import json
import pickle

import numpy as np
from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint, height_to_geopotential
from metpy.units import units
from torch.utils.data import Dataset
from aurora.batch import Batch, Metadata
import torch
import math
import xarray as xr
import zarr

from datetime import datetime, timezone
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# Mapping from Aurora variable names to CERRA dataset variable names.
SURFACE_VAR_MAPPING = {
    "2t" : "t2m",
    "10u" : "10u",
    "10v" : "10v",
    "msl" : "msl",
}

STATIC_VAR_MAPPING = {
    "lsm" : "lsm",
    "slt" : "slt",
    "z" : "zs"
 }

ATMOS_VAR_MAPPING = {
    "t" : "t",
    "u" : "u",
    "v" : "v",
    "q" : "q",
    "z" : "z",
}

# Mapping for the boundary condition variables.
# "unavailable" is used for variables that are not present in the boundary dataset.
BOUNDARY_SURFACE_VAR_MAPPING = {
    "2t" : "2m_temperature",
    "10u" : "10m_u_component_of_wind",
    "10v" : "10m_v_component_of_wind",
    "msl" : "mean_sea_level_pressure",
}

# static vars are not available in IFS-HRES, so we use dummy values for them.
# We could obtain them by interpolating from the static vars in the Aurora repository,
# but for the first version we just use dummy values.
BOUNDARY_STATIC_VAR_MAPPING = {
    "lsm" : "land_sea_mask",
    "slt" : "soil_type",
    "z" : "geopotential_at_surface",
}

BOUNDARY_ATMOS_VAR_MAPPING = {
    "t" : "temperature",
    "u" : "u_component_of_wind",
    "v" : "v_component_of_wind",
    "q" : "specific_humidity",
    "z" : "geopotential",
    "r" : "relative_humidity",
}

class CerraDataset(Dataset):
    """
    For every batch we have current time T and state at T-1. Time dimension is therefore 2.


    """
    def __init__(self,
            data_path: str,
            start_time: str,
            end_time: str,
            surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
            atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
            static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
            normalize: bool = False,
            stats_path: str = None,
            x_range: tuple[int, int] = (0, 1068),
            y_range: tuple[int, int] = (0, 1068),
            boundary_path: str | None = None,
            boundary_size: tuple[int, int, int, int] = (0, 0, 0, 0),  # (left, right, top, bottom)
            rollout_steps: int = 1,
            step_size_hours: int = 24,
            lead_time_hours: int = 24,
            dataset_step_size: int = 24,
            boundary_dataset_step_size: int = 12,
            no_xarray: bool = False,
            use_evaluation_mode: bool = False,
            is_forecast_dataset: bool = False,
            ):
        """
        Args:
            data_path (str): Path to the dataset
            start_time (str): Start time of the dataset, currently only supports 6 AM start time
            end_time (str): End time of the dataset
            normalize (bool): Whether to normalize the dataset
            stats_path (str): Path to the statistics file. Expects a pickled dictionary with variable names as keys and
                tuples of mean and std as values. If None, the dataset will not be normalized!
            x_range (tuple[int, int]): Range of x values to consider
            y_range (tuple[int, int]): Range of y values to consider
            boundary_path (str | None): Path to the dataset containing the boundary conditions. If None, no boundary conditions will be used.
            boundary_size (tuple[int, int, int, int]): Size of the boundary conditions in pixels. Order is (left, right, top, bottom).
            rollout_steps (int): Number of steps to consider in the rollout
            step_size_hours (int): Step size in hours
            lead_time_hours (int): Forecast lead time in hours. Influences the difference between current and previous time.
            dataset_step_size (int): Step size of the dataset
            use_evaluation_mode (bool): Whether to use the dataset in evaluation mode. This means that the dataset will
                only return one timestamp per sample. Not compatible with the no_xarray option.
            is_forecast_dataset (bool): Whether the dataset is a forecast dataset. This means that the dataset will have
                an additional dimension called "prediction_timedelta" which is the time difference between the
                current time and the forecast time. Can only be used in evaluation mode.
        """
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.boundary_size = boundary_size
        self.return_boundaries = boundary_path is not None

        self.ds_step_size_hours = dataset_step_size
        self.boundary_ds_step_size_hours = boundary_dataset_step_size
        self.rollout_steps = rollout_steps
        self.lead_time_hours = lead_time_hours
        self.lead_time_steps = lead_time_hours // self.ds_step_size_hours
        self.step_size = step_size_hours // self.ds_step_size_hours
        self.boundary_step_size = step_size_hours // self.boundary_ds_step_size_hours
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars
        self.normalize = normalize
        self.no_xarray = no_xarray
        self.use_evaluation_mode = use_evaluation_mode
        self.is_forecast_dataset = is_forecast_dataset

        assert not (self.use_evaluation_mode and self.no_xarray), "Evaluation mode is not supported when not using xarray!"
        assert not self.is_forecast_dataset or self.use_evaluation_mode, "Forecast dataset is only supported in evaluation mode!"

        if self.normalize:
            if stats_path is None:
                self.stats = {}
            else:
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)

        if self.no_xarray:
            self._init_zarr(data_path, boundary_path, start_time, end_time)
        else:
            self._init_xarray(data_path, boundary_path, start_time, end_time)

    def _init_zarr(self, data_path: str, boundary_path: str | None, start_time: str, end_time: str):
        self.data = zarr.open(data_path)

        # The time dim doesn't always have the same name, so we accept multiple values
        if "time" in self.data:
            self.time_dim = "time"
        elif "valid_time" in self.data:
            self.time_dim = "valid_time"
        else:
            raise ValueError("No time dimension found for dataset.")

        if "pressure_level" in self.data:
            self.pl_dim = "pressure_level"
        elif "level" in self.data:
            self.pl_dim = "level"
        else:
            raise ValueError("No pressure level dimension found for dataset.")

        # Convert longitude to the range [-180, 180] regardless of whether the dataset is in [0, 360] or [-180, 180]
        longitudes_neg = ((self.data["longitude"][0] % 360) + 180) % 360 - 180

        pressure_level_descending = np.all(self.data[self.pl_dim][:][:-1] > self.data[self.pl_dim][:][1:])
        latitude_ascending = np.all(self.data["latitude"][:, 0][:-1] <= self.data["latitude"][:, 0][1:])
        longitude_ascending = np.all(longitudes_neg[:-1] <= longitudes_neg[1:])
        assert pressure_level_descending, "Use xarray dataset if pressure levels are not descending!"
        assert latitude_ascending, "Use xarray dataset if latitude is not ascending!"
        assert longitude_ascending, "Use xarray dataset if longitude is not ascending!"

        self.longitude = self.data["longitude"][:][::-1]
        self.longitude = self.longitude[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]].copy()
        self.longitude = self.longitude % 360
        self.latitude = self.data["latitude"][:][::-1]
        self.latitude = self.latitude[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]].copy()
        self.pressure_level = self.data[self.pl_dim][:][::-1]
        self.pressure_level = self.pressure_level.copy()

        assert self.longitude.min() >= 0, "Datavar: longitude - Some longitudes are below 0"
        assert self.longitude.max() < 360, "Datavar: longitude - Some longitudes are >= 360"
        assert self.latitude.min() >= 0, "Datavar: latitude - Some latitudes are below 0"
        assert self.latitude.max() < 360, "Datavar: latitude - Some latitudes are >= 360"

        # We changed the default time unit from seconds since 1970-01-01 to nanoseconds since 1970-01-01
        # at some point, so we need to read the metadata to determine the time unit
        metadata_path = data_path + "/.zmetadata"
        is_hf_path = metadata_path.startswith("hf://")
        if is_hf_path:
            from huggingface_hub import hffs
            with hffs.open(metadata_path, "rb") as f:
                metadata = json.load(f)
        else:
            with open(metadata_path, "rb") as f:
                metadata = json.load(f)

        self.time_unit = metadata["metadata"][f"{self.time_dim}/.zattrs"]["units"]
        start_timestamp = int(datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc).timestamp())
        end_timestamp = int(datetime.fromisoformat(end_time).replace(tzinfo=timezone.utc).timestamp())

        if self.time_unit == "nanoseconds since 1970-01-01":
            start_timestamp *= 10**9
            end_timestamp *= 10**9
        elif self.time_unit != "seconds since 1970-01-01":
            raise ValueError("Time unit must be seconds or nanoseconds since 1970-01-01")

        self.start_idx = np.argmax(start_timestamp <= self.data[self.time_dim][:]).item()
        self.end_idx = np.argmax(end_timestamp <= self.data[self.time_dim][:]).item()

        # Make sure that the start and end time are in the dataset
        if start_timestamp < self.data[self.time_dim][:][0]:
            raise ValueError("Start time is not in the dataset. Please check the start time.")
        if end_timestamp > self.data[self.time_dim][:][-1]:
            raise ValueError("End time is not in the dataset. Please check the end time.")

        if boundary_path is not None:
            self.boundary_data = zarr.open(boundary_path)

            # Convert longitude to the range [-180, 180] regardless of whether the dataset is in [0, 360] or [-180, 180]
            longitudes_neg = ((self.boundary_data["longitude"][0] % 360) + 180) % 360 - 180

            pressure_level_descending = np.all(self.boundary_data["pressure_level"][:][:-1] > self.boundary_data["pressure_level"][:][1:])
            latitude_ascending = np.all(self.boundary_data["latitude"][:, 0][:-1] <= self.boundary_data["latitude"][:, 0][1:])
            longitude_ascending = np.all(longitudes_neg[:-1] <= longitudes_neg[1:])
            assert pressure_level_descending, "Use xarray dataset if boundary data pressure levels are not ascending!"
            assert latitude_ascending, "Use xarray dataset if latitude is not ascending!"
            assert longitude_ascending, "Use xarray dataset if longitude is not ascending!"

            # Calculate the slices for the boundary conditions
            # This assumes that the boundary in the boundary dataset has the same size in all dimensions
            b_data_width, b_data_height = self.boundary_data["longitude"][:].shape
            cerra_data_width, cerra_data_height = self.data["longitude"][:].shape
            boundary_size_data = (b_data_width - cerra_data_width) // 2
            x_min = boundary_size_data - self.boundary_size[0] + self.x_range[0]
            x_max = x_min + self.x_range[1] + self.boundary_size[0] + self.boundary_size[1]
            y_min = boundary_size_data - self.boundary_size[2] + self.y_range[0]
            y_max = y_min + self.y_range[1] + self.boundary_size[2] + self.boundary_size[3]

            x_slice = slice(x_min, x_max)
            y_slice = slice(y_min, y_max)
            self.boundary_x_range = (x_min, x_max)
            self.boundary_y_range = (y_min, y_max)

            self.boundary_longitude = self.boundary_data["longitude"][:][::-1]
            self.boundary_longitude = self.boundary_longitude[x_slice, y_slice].copy()
            self.boundary_longitude = self.boundary_longitude.astype(np.float32) % 360
            self.boundary_latitude = self.boundary_data["latitude"][:][::-1]
            self.boundary_latitude = self.boundary_latitude[x_slice, y_slice].astype(np.float32).copy()

            assert self.boundary_longitude.min() >= 0, "Datavar: longitude - Some boundary longitudes are below 0"
            assert self.boundary_longitude.max() < 360, "Datavar: longitude - Some boundary longitudes are >= 360"
            assert self.boundary_latitude.min() >= 0, "Datavar: latitude - Some boundary latitudes are below 0"
            assert self.boundary_latitude.max() < 360, "Datavar: latitude - Some boundary latitudes are >= 360"
            assert np.all(self.pressure_level == self.boundary_data["pressure_level"][:]), "Datavar: pressure level - Pressure levels in boundary data do not match CERRA data"

            # We changed the default time unit from seconds since 1970-01-01 to nanoseconds since 1970-01-01
            # at some point, so we need to read the metadata to determine the time unit
            metadata_path = boundary_path + "/.zmetadata"
            is_hf_path = metadata_path.startswith("hf://")
            if is_hf_path:
                from huggingface_hub import hffs
                with hffs.open(metadata_path, "rb") as f:
                    metadata = json.load(f)
            else:
                with open(metadata_path, "rb") as f:
                    metadata = json.load(f)

            self.boundary_time_unit = metadata["metadata"]["time/.zattrs"]["units"]
            start_timestamp = int(datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc).timestamp())
            end_timestamp = int(datetime.fromisoformat(end_time).replace(tzinfo=timezone.utc).timestamp())

            if self.boundary_time_unit == "nanoseconds since 1970-01-01":
                start_timestamp *= 10 ** 9
                end_timestamp *= 10 ** 9
            elif self.boundary_time_unit != "seconds since 1970-01-01":
                raise ValueError("Time unit must be seconds or nanoseconds since 1970-01-01")

            self.boundary_start_idx = np.argmax(start_timestamp <= self.boundary_data["time"][:]).item()
            self.boundary_end_idx = np.argmax(end_timestamp <= self.boundary_data["time"][:]).item()

            # Make sure that the start and end time are in the dataset
            if start_timestamp < self.boundary_data["time"][:][0]:
                raise ValueError("Start time is not in the dataset. Please check the start time.")
            if end_timestamp > self.boundary_data["time"][:][-1]:
                raise ValueError("End time is not in the dataset. Please check the end time.")


    def _init_xarray(self, data_path: str, boundary_path: str | None, start_time: str, end_time: str):
        self.data = xr.open_zarr(data_path)

        # The time dim doesn't always have the same name, so we accept multiple values
        if "time" in self.data:
            self.time_dim = "time"
        elif "valid_time" in self.data:
            self.time_dim = "valid_time"
        else:
            raise ValueError("No time dimension found for dataset.")

        if np.datetime64(start_time) < self.data[self.time_dim].min():
            raise ValueError("Start time is not in the dataset. Please check the end time.")
        if np.datetime64(end_time) > self.data[self.time_dim].max():
            raise ValueError("End time is not in the dataset. Please check the end time.")

        if "pressure_level" in self.data:
            self.pl_dim = "pressure_level"
        elif "level" in self.data:
            self.pl_dim = "level"
        else:
            raise ValueError("No pressure level dimension found for dataset.")

        if self.is_forecast_dataset:
            # When dealing with a forecast dataset, we need to also select the lead time
            lead_time_hours = self.lead_time_steps * self.ds_step_size_hours
            lead_time_timedelta = np.timedelta64(lead_time_hours, "h")
            self.data = self.data.sel(prediction_timedelta=lead_time_timedelta)

        # Ensure the correct order of dimensions
        self.data = self.data.transpose(self.time_dim, self.pl_dim, "y", "x")
        lon_for_sorting = ((self.data.longitude[0] + 180) % 360) - 180  # Ensure longitude is in [-180, 180] for sorting
        self.data = self.data.sortby([-self.data.latitude[:, 0], lon_for_sorting, self.data[self.pl_dim]])

        # Ensure longitude is in [0, 360] for Aurora
        # The conversion to float32 is necessary to avoid issues with the type conversion in the training loop
        # Without it, it might be the case that the longitude values are not in the correct range after conversion
        self.data["longitude"].values = self.data["longitude"].values.astype(np.float32) % 360

        assert (self.data["longitude"].min(axis=1) >= 0).all(), "Datavar: longitude - Some longitudes are below 0"
        assert (self.data["longitude"].min(axis=1) < 360).all(), "Datavar: longitude - Some longitudes are >= 360"
        assert (self.data["latitude"].min(axis=1) >= 0).all(), "Datavar: latitude - Some longitudes are below 0"
        assert (self.data["latitude"].min(axis=1) < 360).all(), "Datavar: latitude - Some longitudes are >= 360"

        time_slice = slice(start_time, end_time)
        x_slice = slice(self.x_range[0], self.x_range[1])
        y_slice = slice(self.y_range[0], self.y_range[1])

        # Select the data for the given time range and x/y ranges
        cerra_data_width, cerra_data_height = self.data.latitude.shape
        selection = {self.time_dim: time_slice, "x": x_slice, "y": y_slice}
        self.data = self.data.sel(selection)

        if boundary_path is not None:
            # Load the boundary conditions dataset
            self.boundary_data = xr.open_zarr(boundary_path)
            self.boundary_data = self.boundary_data.transpose("time", "prediction_timedelta", "pressure_level", "y", "x")

            lon_for_sorting = ((self.boundary_data.longitude[0] + 180) % 360) - 180  # Ensure longitude is in [-180, 180] for sorting
            self.boundary_data = self.boundary_data.sortby([-self.boundary_data.latitude[:, 0], lon_for_sorting, self.boundary_data["pressure_level"]])

            # Ensure longitude is in [0, 360] for Aurora
            # The conversion to float32 is necessary to avoid issues with the type conversion in the training loop
            # Without it, it might be the case that the longitude values are not in the correct range after conversion
            self.boundary_data["longitude"].values = self.boundary_data["longitude"].values.astype(np.float32) % 360

            assert (self.boundary_data["longitude"].min(axis=1) >= 0).all(), "Datavar: longitude - Some longitudes are below 0"
            assert (self.boundary_data["longitude"].min(axis=1) < 360).all(), "Datavar: longitude - Some longitudes are >= 360"
            assert (self.boundary_data["latitude"].min(axis=1) >= 0).all(), "Datavar: latitude - Some longitudes are below 0"
            assert (self.boundary_data["latitude"].min(axis=1) < 360).all(), "Datavar: latitude - Some longitudes are >= 360"

            # Calculate the slices for the boundary conditions
            # This assumes that the boundary in the boundary dataset has the same size in all dimensions
            b_data_width, b_data_height = self.boundary_data.latitude.shape
            boundary_size_data = (b_data_width - cerra_data_width) // 2
            x_min = boundary_size_data - self.boundary_size[0] + self.x_range[0]
            x_max = x_min + self.x_range[1] + self.boundary_size[0] + self.boundary_size[1]
            y_min = boundary_size_data - self.boundary_size[2] + self.y_range[0]
            y_max = y_min + self.y_range[1] + self.boundary_size[2] + self.boundary_size[3]

            x_slice = slice(x_min, x_max)
            y_slice = slice(y_min, y_max)
            boundary_selection = {self.time_dim: time_slice, "x": x_slice, "y": y_slice}
            self.boundary_data = self.boundary_data.sel(boundary_selection)

    def _get_sample_zarr(
            self,
            id_range: list[int],
            var: str,
            is_boundary: bool,
            timedelta_id: int = None,
    ) -> torch.Tensor:
        if is_boundary:
            data = self.boundary_data
            x_slice = slice(self.boundary_x_range[0], self.boundary_x_range[1])
            y_slice = slice(self.boundary_y_range[0], self.boundary_y_range[1])
            surface_slice = (slice(None), timedelta_id, x_slice, y_slice)
            atmos_slice = (slice(None), timedelta_id, slice(None), x_slice, y_slice)
            static_slice = (x_slice, y_slice)
        else:
            data = self.data
            x_slice = slice(self.x_range[0], self.x_range[1])
            y_slice = slice(self.y_range[0], self.y_range[1])
            surface_slice = (slice(None), x_slice, y_slice)
            atmos_slice = (slice(None), slice(None), x_slice, y_slice)
            static_slice = (x_slice, y_slice)

        # Compute derived variables online if not precomputed
        if (var == "10v" or var == "10u") and var not in data:
            assert not is_boundary, "10u and 10v variables are not available in the boundary dataset"
            wind_speed = data["si10"][id_range][:, ::-1]
            wdir = data["wdir10"][id_range][:, ::-1]
            wind_speed = wind_speed[surface_slice][None]
            wdir = wdir[surface_slice][None]

            wdir_rad = np.radians(wdir)

            if var == "10v":
                v = -wind_speed * np.cos(wdir_rad)
                return torch.from_numpy(v.copy())
            else:
                u = -wind_speed * np.sin(wdir_rad)
                return torch.from_numpy(u.copy())

        elif var == "q" and var not in data:
            assert not is_boundary, "q is not available in the boundary dataset"

            temperature = data["t"][id_range][:, ::-1, ::-1]
            relative_h = data["r"][id_range][:, ::-1, ::-1]
            temperature = temperature[atmos_slice][None]
            relative_h = relative_h[atmos_slice][None]
            pressure_levels = self.pressure_level
            # Shape: (pressure_level,)
            pressure = np.expand_dims(pressure_levels, axis=(0, -1, -2))  # Initial shape: (1, pressure_level, 1, 1)
            pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (1, pressure_level, lat, lon)
            pressure = pressure * units.hPa
            temperature_k = temperature * units.kelvin
            relative_h = np.maximum(relative_h, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
            dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h)
            specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

            return torch.from_numpy(specific_humidity.m.copy())

        elif var == "zs" and var not in data:
            assert not is_boundary, "zs is not available in the boundary dataset"
            orography = data["orog"][id_range][:, ::-1]
            orography = orography[surface_slice]
            orography = orography * units.m
            geopotential = height_to_geopotential(orography)

            return torch.from_numpy(geopotential.m.copy())

        elif var == "unavailable_static":
            assert is_boundary, "unavailable_static variable is only available in the boundary dataset"
            data_point = data["2m_temperature"][id_range[0]][::-1]
            data_point = data_point[static_slice]
            shape = data_point.shape
            return torch.zeros(shape)

        elif var in STATIC_VAR_MAPPING.values() or var in BOUNDARY_STATIC_VAR_MAPPING.values():
            # The static vars may not have a time dimension (which they don't need since they are constant)
            if len(data[var].shape) == 2:
                data_point = data[var][:][::-1]
            else:
                data_point = data[var][id_range[0]][::-1]
            data_point = data_point[static_slice]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

        elif var in SURFACE_VAR_MAPPING.values() or var in BOUNDARY_SURFACE_VAR_MAPPING.values():
            if is_boundary:
                data_point = data[var][id_range][:, :, ::-1]
            else:
                data_point = data[var][id_range][:, ::-1]
            data_point = data_point[surface_slice][None]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

        else:
            if is_boundary:
                data_point = data[var][id_range][:, :, :, ::-1]
            else:
                data_point = data[var][id_range][:, ::-1, ::-1]
            data_point = data_point[atmos_slice][None]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

    def _get_metadata_zarr(self, id_range: list[int], is_boundary: bool) -> Metadata:
        if is_boundary:
            data = self.boundary_data
            time_dim = "time"
            time_unit = self.boundary_time_unit
            latitude = self.boundary_latitude
            longitude = self.boundary_longitude
        else:
            data = self.data
            time_dim = self.time_dim
            time_unit = self.time_unit
            latitude = self.latitude
            longitude = self.longitude


        time = data[time_dim][id_range]

        if time_unit == "nanoseconds since 1970-01-01":
            time = time.astype("int64") // 10**9

        # Select only the last timestamp in accordance with the batch specification at
        # https://microsoft.github.io/aurora/batch.html#batch-metadata
        time_metadata = [datetime.fromtimestamp(t, timezone.utc) for t in time]
        time_metadata = time_metadata[-1],

        levels = tuple(self.pressure_level.astype(int))

        return Metadata(
            lat=torch.from_numpy(latitude),
            lon=torch.from_numpy(longitude),
            time=time_metadata,
            atmos_levels=levels
        )

    def _get_sample_xarray(self, selection: xr.Dataset, var: str) -> torch.Tensor:
        # Compute derived variables online if not precomputed
        if (var == "10v" or var == "10u") and var not in self.data:
            wind_speed = selection["si10"].values[None]
            wdir = selection["wdir10"].values[None]
            wdir_rad = np.radians(wdir)

            if var == "10v":
                v = -wind_speed * np.cos(wdir_rad)
                return torch.from_numpy(v)
            else:
                u = -wind_speed * np.sin(wdir_rad)
                return torch.from_numpy(u)

        elif var == "q" and var not in self.data.data_vars:
            temperature = selection["t"].values[None]
            relative_h = selection["r"].values[None]
            pressure_levels = self.data.coords[self.pl_dim].values
            # Shape: (pressure_level,)
            pressure = np.expand_dims(pressure_levels, axis=(0, -1, -2))  # Initial shape: (1, pressure_level, 1, 1)
            pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (1, pressure_level, lat, lon)
            pressure = pressure * units.hPa
            temperature_k = temperature * units.kelvin
            relative_h = np.maximum(relative_h, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
            dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h)
            specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

            return torch.from_numpy(specific_humidity.m)

        elif var == "zs" and var not in self.data.data_vars:
            orography = selection["orog"].values[None]
            orography = orography * units.m
            geopotential = height_to_geopotential(orography)

            return torch.from_numpy(geopotential.m.copy())

        elif var == "unavailable_static":
            shape = selection["2m_temperature"].values.shape
            return torch.zeros(shape)

        elif var in STATIC_VAR_MAPPING.values() or var in BOUNDARY_STATIC_VAR_MAPPING.values():
            data_point = selection[var].values
            data_point = torch.from_numpy(data_point)
            return data_point

        else:
            data_point = selection[var].values[None]
            data_point = torch.from_numpy(data_point)

            return data_point

    @staticmethod
    def _get_metadata_xarray(selection: xr.Dataset) -> Metadata:
        # We don't know which dataset the selection is from, so we have to check the dimensions
        if "time" in selection:
            time_dim = "time"
        elif "valid_time" in selection:
            time_dim = "valid_time"
        else:
            raise ValueError("No time dimension found for dataset.")
        if "pressure_level" in selection:
            pl_dim = "pressure_level"
        elif "level" in selection:
            pl_dim = "level"
        else:
            raise ValueError("No pressure level dimension found for dataset.")

        # Select only the last timestamp in accordance with the batch specification at
        # https://microsoft.github.io/aurora/batch.html#batch-metadata
        time = selection[time_dim].values
        time_metadata = [datetime.fromtimestamp(t.astype("datetime64[s]").astype("int"), timezone.utc) for t in time]
        time_metadata = time_metadata[-1],

        levels = tuple(selection[pl_dim].values.astype(int))

        return Metadata(
            lat=torch.from_numpy(selection.latitude.values),
            lon=torch.from_numpy(selection.longitude.values),
            time=time_metadata,
            atmos_levels=levels
        )

    def __len__(self) -> int:
        # In evaluation mode, we only return one sample per time step
        if self.use_evaluation_mode:
            return len(self.data[self.time_dim])

        # In training mode we return (2 + self.rollout_steps) values per index, so we
        # have to subtract (1 + self.rollout_steps) * self.lead_time_steps from the number of timesteps
        if self.no_xarray:
            num_steps = (1 + self.end_idx - self.start_idx) - (1 + self.rollout_steps) * self.lead_time_steps
        else:
            num_steps = len(self.data[self.time_dim]) - (1 + self.rollout_steps) * self.lead_time_steps
        length = math.ceil(num_steps / self.step_size)

        return length

    def __getitem__(self, idx: int) -> dict[str, Batch]:
        if self.use_evaluation_mode:
            batch = self._create_batch_xarray([idx])

            return {"input": batch}

        # step size -> time delta between two steps in the dataset
        # lead time is the time between prev_idx and current idx
        # it is also the diff between t+1 and t.
        if self.no_xarray:
            prev_idx = self.start_idx + idx * self.step_size
            current_idx = prev_idx + self.lead_time_steps
            target_idx = current_idx + self.lead_time_steps

            if self.return_boundaries:
                current_boundary_idx = self.boundary_start_idx + idx * self.boundary_step_size
                data_time = self.data["time"][:][current_idx]
                boundary_time = self.boundary_data["time"][:][current_boundary_idx]

                if self.time_unit == "nanoseconds since 1970-01-01":
                    data_time = int(data_time / 10 ** 9)
                if self.boundary_time_unit == "nanoseconds since 1970-01-01":
                    boundary_time = int(boundary_time / 10 ** 9)

                assert data_time == boundary_time, "Data time and boundary time do not match!"
        else:
            prev_idx = idx * self.step_size
            current_idx = prev_idx + self.lead_time_steps
            target_idx = current_idx + self.lead_time_steps

            if self.return_boundaries:
                current_boundary_idx = idx * self.boundary_step_size
                data_time = self.data[self.time_dim][current_idx]
                boundary_time = self.boundary_data[self.time_dim][current_boundary_idx]

                assert data_time == boundary_time, "Data time and boundary time do not match!"

        output = {}

        if self.no_xarray:
            output["input"] = self._create_batch_zarr([prev_idx, current_idx])

            if self.return_boundaries:
                # The IFS-HRES forecast is only available in 12-hour intervals
                # so we can only return the boundary conditions for the current step
                # and not for the previous step.
                boundary = self._create_boundary_batch_zarr([current_boundary_idx], 0)
        else:
            output["input"] = self._create_batch_xarray([prev_idx, current_idx])

            if self.return_boundaries:
                # The IFS-HRES forecast is only available in 12-hour intervals
                # so we can only return the boundary conditions for the current step
                # and not for the previous step.
                boundary = self._create_boundary_batch_xarray([current_boundary_idx], 0)

        if self.return_boundaries:
            # Add a second, zeroed out boundary sample
            for k in boundary.surf_vars:
                second_sample_var = torch.zeros_like(boundary.surf_vars[k])
                boundary.surf_vars[k] = torch.cat([second_sample_var, boundary.surf_vars[k]], dim=1)

            for k in boundary.atmos_vars:
                second_sample_var = torch.zeros_like(boundary.atmos_vars[k])
                boundary.atmos_vars[k] = torch.cat([second_sample_var, boundary.atmos_vars[k]], dim=1)

            output["input_boundary"] = boundary

        for i in range(0, self.rollout_steps):
            if self.no_xarray:
                output[f"target_{i}"] = self._create_batch_zarr([target_idx])

                # Return boundaries for each target step if required
                # We don't need to return boundaries for the last target step as we do not have a next step to predict
                if self.return_boundaries and i < self.rollout_steps - 1:
                    lead_time = (i + 1) * self.lead_time_hours
                    output[f"input_boundary_{i}"] = self._create_boundary_batch_zarr([current_boundary_idx], lead_time)
            else:
                output[f"target_{i}"] = self._create_batch_xarray([target_idx])

                # Return boundaries for each target step if required
                # We don't need to return boundaries for the last target step as we do not have a next step to predict
                if self.return_boundaries and i < self.rollout_steps - 1:
                    lead_time = (i + 1) * self.lead_time_hours
                    output[f"input_boundary_{i}"] = self._create_boundary_batch_xarray([current_boundary_idx], lead_time)

            target_idx += self.lead_time_steps

        return output

    def _create_batch_zarr(self, id_range: list[int]) -> Batch:
        mapped_surf_vars = {k: SURFACE_VAR_MAPPING[k] for k in self.surf_vars}
        mapped_atmos_vars = {k: ATMOS_VAR_MAPPING[k] for k in self.atmos_vars}
        mapped_static_vars = {k: STATIC_VAR_MAPPING[k] for k in self.static_vars}

        surf_vars = {k : self._get_sample_zarr(id_range, v, is_boundary=False) for k, v in mapped_surf_vars.items()}
        atmos_vars = {k : self._get_sample_zarr(id_range, v, is_boundary=False) for k, v in mapped_atmos_vars.items()}
        static_vars = {k : self._get_sample_zarr(id_range, v, is_boundary=False) for k, v in mapped_static_vars.items()}
        meta = self._get_metadata_zarr(id_range, is_boundary=False)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch

    def _create_boundary_batch_zarr(self, id_range: list[int], lead_time_hours: int) -> Batch:
        timedelta_id = np.nonzero(self.boundary_data["prediction_timedelta"][:] == lead_time_hours)[0].item()
        mapped_surf_vars = {k: BOUNDARY_SURFACE_VAR_MAPPING[k] for k in self.surf_vars}
        mapped_atmos_vars = {k: BOUNDARY_ATMOS_VAR_MAPPING[k] for k in self.atmos_vars}
        mapped_static_vars = {k: BOUNDARY_STATIC_VAR_MAPPING[k] for k in self.static_vars}

        surf_vars = {k: self._get_sample_zarr(id_range, v, is_boundary=True, timedelta_id=timedelta_id) for k, v in mapped_surf_vars.items()}
        atmos_vars = {k: self._get_sample_zarr(id_range, v, is_boundary=True, timedelta_id=timedelta_id) for k, v in mapped_atmos_vars.items()}
        static_vars = {k: self._get_sample_zarr(id_range, v, is_boundary=True, timedelta_id=timedelta_id) for k, v in mapped_static_vars.items()}
        meta = self._get_metadata_zarr(id_range, is_boundary=True)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch

    def _create_batch_xarray(self, id_range: list[int]) -> Batch:

        selection = self.data.isel({self.time_dim: id_range})
        selection_static = self.data.isel({self.time_dim: id_range[0]})
        mapped_surf_vars = {k: SURFACE_VAR_MAPPING[k] for k in self.surf_vars}
        mapped_atmos_vars = {k: ATMOS_VAR_MAPPING[k] for k in self.atmos_vars}
        mapped_static_vars = {k: STATIC_VAR_MAPPING[k] for k in self.static_vars}

        surf_vars = {k : self._get_sample_xarray(selection, v) for k, v in mapped_surf_vars.items()}
        atmos_vars = {k : self._get_sample_xarray(selection, v) for k, v in mapped_atmos_vars.items()}
        static_vars = {k : self._get_sample_xarray(selection_static, v) for k, v in mapped_static_vars.items()}
        meta = self._get_metadata_xarray(selection)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch

    def _create_boundary_batch_xarray(self, id_range: list[int], lead_time_hours: int) -> Batch:

        selection = self.boundary_data.isel({self.time_dim: id_range})
        selection = selection.sel(prediction_timedelta=np.timedelta64(lead_time_hours, "h"))
        selection_static = self.boundary_data.isel({self.time_dim: id_range[0]})
        selection_static = selection_static.sel(prediction_timedelta=np.timedelta64(lead_time_hours, "h"))
        mapped_surf_vars = {k: BOUNDARY_SURFACE_VAR_MAPPING[k] for k in self.surf_vars}
        mapped_atmos_vars = {k: BOUNDARY_ATMOS_VAR_MAPPING[k] for k in self.atmos_vars}
        mapped_static_vars = {k: BOUNDARY_STATIC_VAR_MAPPING[k] for k in self.static_vars}

        surf_vars = {k : self._get_sample_xarray(selection, v) for k, v in mapped_surf_vars.items()}
        atmos_vars = {k : self._get_sample_xarray(selection, v) for k, v in mapped_atmos_vars.items()}
        static_vars = {k : self._get_sample_xarray(selection_static, v) for k, v in mapped_static_vars.items()}
        meta = self._get_metadata_xarray(selection)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch
