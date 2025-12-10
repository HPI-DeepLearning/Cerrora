import itertools
import logging
import time
import multiprocessing as mp
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from tqdm import tqdm

LOG = logging.getLogger(__name__)

class CerraRegionCropper:
    def __init__(
            self,
            aurora_lons_lats_path: str,
            boundary_width: tuple[int, int, int, int] = (150, 150, 150, 150),
            num_cores: int | None = None,
    ):
        LOG.info("Initializing CERRA region cropper...")
        self.aurora_lons_lats_path = aurora_lons_lats_path
        self.boundary_width = boundary_width  # (left, right, top, bottom)
        self.num_cores = num_cores if num_cores is not None else mp.cpu_count()

        self.lon_lat_buffer = 0.25 # in degrees, needed to avoid nans on the edges of the grid and to have more data points for interpolation
        self.buffer_left_right_top_bottom = (0.25, 0.25, 0.25, 0.25) # add a margin to avoid having no values at the top for interpolation

        self.cerra_resolution =  5500 # Target resolution in meters
        self.globe = ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229) 
        self.lambert_projection = ccrs.LambertConformal( # Projection from PlateCarree to LambertConformal
            central_longitude=8,
            central_latitude=50,
            standard_parallels=(50, 50),
            globe=self.globe,
        )
        self.mercator_projection = ccrs.PlateCarree(globe=self.globe) # Reverse projection (LambertConformal -> PlateCarree)

        self._init_xy_cerra_grid()

    
    def crop_and_interpolate(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Crop ERA5 data to the CERRA grid's region and interpolate it onto the CERRA's grid.
        """
        LOG.info("Starting cropping...")
        start_time = time.time()

        dataset = dataset.sel(pressure_level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        dataset = self._map_dataset_longitude_coords_to_180(dataset)

        # 1. Crop the dataset to the CERRA region in lat/lon space to ensure that all data within the CERRA region is contained
        min_lat = self.lat_cerra_grid.min()
        max_lat = self.lat_cerra_grid.max()
        min_lon = self.lon_cerra_grid.min()
        max_lon = self.lon_cerra_grid.max()

        buffer_left, buffer_right, buffer_top, buffer_bottom = self.buffer_left_right_top_bottom
        mask = (
            (dataset['latitude'] <= max_lat + buffer_top) & (dataset['latitude'] >= min_lat - buffer_bottom) &
            (dataset['longitude'] >= min_lon - buffer_left) & (dataset['longitude'] <= max_lon + buffer_right)
        ).compute()
        cropped_data = dataset.where(mask, drop=True)

        # 2. Convert the dataset coordinates into the Lambert Conformal space 
        cropped_data = self._project_dataset_coords_to_cerra(cropped_data)
        xy_dataset_points = np.array([cropped_data['projected_x'].values.ravel(), cropped_data['projected_y'].values.ravel()]).T

        # 3. Create a new dataset with the interpolated data and the CERRA grid coordinates
        interpolated_dataset = xr.Dataset(
            {},
            coords={
                'time': cropped_data['time'].values,
                'prediction_timedelta': cropped_data['prediction_timedelta'].values,
                'pressure_level': cropped_data['pressure_level'].values,
                'latitude': (('y', 'x'), self.lat_cerra_grid),
                'longitude': (('y', 'x'), self.lon_cerra_grid),
            }
        )

        # 4. Interpolate the dataset onto the already initialized CERRA grid
        LOG.info(f"Starting parallel interpolation on {self.num_cores} cores...")
        for var in cropped_data.data_vars:
            var_data = cropped_data[var].values
            interpolated_dataset = self._initialize_variable_array(var, var_data, interpolated_dataset)

            # For static vars, we can skip the time and timedelta dimensions
            if var_data.ndim == 2:  # (y, x)
                time_idx, timedelta_idx = 0, 0
                result = self._interpolate_time_timedelta((var, var_data, xy_dataset_points, time_idx, timedelta_idx))
                interpolated_dataset[var] = ("y", "x"), result[2]
                continue

            num_timesteps = var_data.shape[0]
            num_timedeltas = var_data.shape[1]
            combinations = itertools.product(range(num_timesteps), range(num_timedeltas))
            # Prepare arguments for parallel processing
            args_list = [(var, var_data, xy_dataset_points, time_idx, leadtime_idx) for time_idx, leadtime_idx in combinations]

            with mp.Pool(self.num_cores) as pool:  # parallelize computation
                for time_idx, timedelta_idx, result in pool.imap_unordered(self._interpolate_time_timedelta, args_list):
                    interpolated_dataset[var][time_idx, timedelta_idx] = result

        interpolated_dataset = self._reverse_map_dataset_longitude_coords_to_360(interpolated_dataset)

        LOG.info(f"Done. Time taken for cropping and interpolation: {time.time() - start_time:.2f} seconds.")
        return interpolated_dataset

    def _append_boundaries(self, coord_grid: np.ndarray, is_x_coords: bool) -> np.ndarray:
        # For the x coordinates, the values differ in the horizontal direction,
        # while for the y coordinates they differ in the vertical direction
        if is_x_coords:
            horizontal_offset = self.cerra_resolution
            vertical_offset = 0
        else:
            horizontal_offset = 0
            vertical_offset = self.cerra_resolution

        # Append the boundary coordinates to the CERRA grid
        num_reps_left = np.arange(self.boundary_width[0])[::-1] + 1
        offset_left = num_reps_left * horizontal_offset
        boundary_left = np.repeat(coord_grid[:, 0:1], self.boundary_width[0], 1) - offset_left

        num_reps_right = np.arange(self.boundary_width[1]) + 1
        offset_right = num_reps_right * horizontal_offset
        boundary_right = np.repeat(coord_grid[:, -1:], self.boundary_width[1], 1) + offset_right

        coord_grid = np.concatenate((boundary_left, coord_grid, boundary_right), axis=1)

        num_reps_top = np.arange(self.boundary_width[2])[::-1] + 1
        offset_top = num_reps_top * vertical_offset
        boundary_top = np.repeat(coord_grid[0:1, :], self.boundary_width[2], 0) - offset_top[:, np.newaxis]

        num_reps_bottom = np.arange(self.boundary_width[3]) + 1
        offset_bottom = num_reps_bottom * vertical_offset
        boundary_bottom = np.repeat(coord_grid[-1:, :], self.boundary_width[3], 0) + offset_bottom[:, np.newaxis]

        coord_grid = np.concatenate((boundary_top, coord_grid, boundary_bottom), axis=0)

        return coord_grid

    def _init_xy_cerra_grid(self):
        """
        Initialize the CERRA grid and the CERRA grid coordinates for the new dataset.
        Calculated during initialization to avoid recalculating the grid every time.
        """
        lon_lat = np.load(self.aurora_lons_lats_path)
        self.lon_cerra_grid = lon_lat['lon']
        self.lat_cerra_grid = lon_lat['lat']

        # This is not necessarily needed (because the projection handles coordinates in range (-180, 180) and (0, 360)) but it's done here for consistency
        self.lon_cerra_grid = np.where(self.lon_cerra_grid > 180, self.lon_cerra_grid - 360, self.lon_cerra_grid)

        lon_lat_points = np.array([self.lon_cerra_grid.ravel(), self.lat_cerra_grid.ravel()]).T
        projected_points = np.array([
            self.lambert_projection.transform_point(lon, lat, self.mercator_projection)
            for lon, lat in lon_lat_points
        ])

        nrows, ncols = self.lat_cerra_grid.shape # longitude and latitude have the same shape
        self.x_cerra_grid = projected_points[:, 0].reshape(nrows, ncols)
        self.y_cerra_grid = projected_points[:, 1].reshape(nrows, ncols)

        self.x_cerra_grid = self._append_boundaries(self.x_cerra_grid, is_x_coords=True)
        self.y_cerra_grid = self._append_boundaries(self.y_cerra_grid, is_x_coords=False)

        # Project the coordinates back to the Mercator projection to update self.lon_cerra_grid and self.lat_cerra_grid
        x_y_points = np.array([self.x_cerra_grid.ravel(), self.y_cerra_grid.ravel()]).T
        projected_points = np.array([
            self.mercator_projection.transform_point(x, y, self.lambert_projection)
            for x, y in x_y_points
        ])

        nrows, ncols = self.x_cerra_grid.shape  # lon / lat shapes have changed after appending boundaries
        self.lon_cerra_grid = projected_points[:, 0].reshape(nrows, ncols)
        self.lat_cerra_grid = projected_points[:, 1].reshape(nrows, ncols)

    def _map_dataset_longitude_coords_to_180(self, dataset: xr.Dataset) -> xr.Dataset:
        longitudes = dataset['longitude'].values
        longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
        dataset = dataset.assign_coords(longitude=(("y", "x"), longitudes))
        return dataset

    def _reverse_map_dataset_longitude_coords_to_360(self, dataset: xr.Dataset) -> xr.Dataset:
        longitudes = dataset['longitude'].values
        longitudes = np.where(longitudes < 0, longitudes + 360, longitudes)
        dataset = dataset.assign_coords(longitude=(("y", "x"), longitudes))
        return dataset

    def _project_dataset_coords_to_cerra(self, dataset: xr.Dataset) -> xr.Dataset:
        lon_lat_points = np.array([dataset['longitude'].values.ravel(), dataset['latitude'].values.ravel()]).T

        projected_points = np.array([
            self.lambert_projection.transform_point(lon, lat, self.mercator_projection)
            for lon, lat in lon_lat_points
        ])

        nrows, ncols = dataset['latitude'].shape # longitude and latitude have the same shape
        projected_x = projected_points[:, 0].reshape(nrows, ncols)
        projected_y = projected_points[:, 1].reshape(nrows, ncols)
        dataset = dataset.assign_coords(
            projected_x=(("y", "x"), projected_x),
            projected_y=(("y", "x"), projected_y)
        )

        return dataset

    def _initialize_variable_array(self, var: str, var_data: np.ndarray, interpolated_dataset: xr.Dataset) -> None:
        num_timesteps = len(interpolated_dataset["time"])
        num_timedeltas = len(interpolated_dataset["prediction_timedelta"])
        num_x = len(interpolated_dataset["x"])
        num_y = len(interpolated_dataset["y"])

        if var_data.ndim == 5: # (time, timedelta, level, y, x)
            num_pressure_levels = len(interpolated_dataset["pressure_level"])
            interpolated_dataset[var] = (
                ("time", "prediction_timedelta", "pressure_level", "y", "x"),
                np.empty((num_timesteps, num_timedeltas, num_pressure_levels, num_y, num_x), dtype=var_data.dtype),
            )
        elif var_data.ndim == 4: # (time, timedelta, y, x)
            interpolated_dataset[var] = (
                ("time", "prediction_timedelta", "y", "x"),
                np.empty((num_timesteps, num_timedeltas, num_y, num_x), dtype=var_data.dtype),
            )
        elif var_data.ndim == 2: # (y, x)
            interpolated_dataset[var] = (
                ("y", "x"),
                np.empty((num_y, num_x), dtype=var_data.dtype),
            )
        else:
            raise ValueError(f"Unsupported variable shape for {var}: {interpolated_dataset[var].shape}")
        return interpolated_dataset

    def _interpolate_time_timedelta(self, args):
        var, var_data, xy_dataset_points, time_idx, timedelta_idx = args  # var_data does have a time dimension!
        method = "nearest" if var == "soil_type" else "linear"  # Soil type is categorical, use nearest neighbor interpolation

        if var_data.ndim == 5:  # (time, timedelta, level, y, x)
            result = np.empty((var_data.shape[2], self.lat_cerra_grid.shape[0], self.lat_cerra_grid.shape[1]),
                              dtype=var_data.dtype)
            for level_idx in range(var_data.shape[2]):
                source_values = var_data[time_idx, timedelta_idx, level_idx].ravel()
                interpolated_values = griddata(
                    xy_dataset_points, source_values,
                    (self.x_cerra_grid, self.y_cerra_grid),
                    method=method
                )
                result[level_idx] = interpolated_values
        elif var_data.ndim == 4:  # (time, timedelta, y, x)
            source_values = var_data[time_idx, timedelta_idx].ravel()
            interpolated_values = griddata(
                xy_dataset_points, source_values,
                (self.x_cerra_grid, self.y_cerra_grid),
                method=method
            )
            result = interpolated_values
        else:  # (y, x)
            source_values = var_data.ravel()
            interpolated_values = griddata(
                xy_dataset_points, source_values,
                (self.x_cerra_grid, self.y_cerra_grid),
                method=method
            )
            result = interpolated_values

        return time_idx, timedelta_idx, result

