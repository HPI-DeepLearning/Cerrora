"""Precompute derived variables for CERRA dataset."""
import argparse
import os

import numpy as np
from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint, height_to_geopotential
from metpy.units import units
from numcodecs import Blosc
from tqdm import tqdm

from main import check_and_start_debugger
import xarray as xr


def compute_wind_components(wind_speed: np.ndarray, wind_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wind_direction_rad = np.radians(wind_direction)
    v = -wind_speed * np.cos(wind_direction_rad)
    u = -wind_speed * np.sin(wind_direction_rad)

    return u, v


def compute_specific_humidity(temperature: np.ndarray, relative_humidity: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    pressure = np.expand_dims(pressure, axis=(-1, -2))  # Initial shape: (pressure_level, 1, 1)
    pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (pressure_level, lat, lon)
    pressure = pressure * units.hPa
    temperature_k = temperature * units.kelvin
    relative_h = np.maximum(relative_humidity, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
    dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h)
    specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

    return specific_humidity.magnitude

def compute_geopotential_at_surface(orography: np.ndarray) -> np.ndarray:
    height = orography * units.m
    geopotential = height_to_geopotential(height)

    return geopotential.magnitude

def main(args: argparse.Namespace):
    """Compute 10m u/v wind components from 10m wind speed and direction.
    In addition, compute specific humidity from relative humidity and temperature and pressure.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    chunks = {"time": 1, "pressure_level": 13, "y": 1069, "x": 1069}
    data = xr.open_zarr(args.src_dir, chunks=chunks)

    is_forecast_dataset = "prediction_timedelta" in data

    if "pressure_level" in data:
        pl_dim = "pressure_level"
    else:
        pl_dim = "level"

    if args.start_date is not None and args.end_date is not None:
        data = data.sel(time=slice(args.start_date, args.end_date))

    wind_u_values = []
    wind_v_values = []
    specific_humidity_values = []
    geopotential_values = []

    for i in tqdm(range(len(data["time"]))):
        sample = data.isel(time=i)

        if args.compute_10m_wind:
            wind_speed = sample["si10"].values
            wind_direction = sample["wdir10"].values

            u, v = compute_wind_components(wind_speed, wind_direction)
            wind_u_values.append(u)
            wind_v_values.append(v)

        if args.compute_specific_humidity:
            temperature = sample["t"].values
            relative_humidity = sample["r"].values
            pressure = sample[pl_dim].values

            specific_humidity = compute_specific_humidity(temperature, relative_humidity, pressure)
            specific_humidity_values.append(specific_humidity)

        if args.compute_geopotential_at_surface:
            orography = sample["orog"].values
            geopotential = compute_geopotential_at_surface(orography)
            geopotential_values.append(geopotential)

        if (i + 1) % args.save_freq == 0 or i == len(data["time"]) - 1:
            start_idx = i - (i % args.save_freq)  # This ensures that even the last chunk is saved correctly
            data_subset = data.isel(time=slice(start_idx, i + 1))

            if is_forecast_dataset:
                wind_dims = ["time", "prediction_timedelta", "y", "x"]
                q_dims = ["time", "prediction_timedelta", pl_dim, "y", "x"]
                wind_coords = {"time": data_subset["time"], "prediction_timedelta": data_subset["prediction_timedelta"]}
                q_coords = {"time": data_subset["time"], "prediction_timedelta": data_subset["prediction_timedelta"], pl_dim: data[pl_dim]}
                wind_chunks = {"time": 1, "prediction_timedelta": 1, "y": 1069, "x": 1069}
                q_chunks = {"time": 1, "prediction_timedelta": 1, pl_dim: 13, "y": 1069, "x": 1069}
            else:
                wind_dims = ["time", "y", "x"]
                q_dims = ["time", pl_dim, "y", "x"]
                wind_coords = {"time": data_subset["time"]}
                q_coords = {"time": data_subset["time"], pl_dim: data[pl_dim]}
                wind_chunks = {"time": 1, "y": 1069, "x": 1069}
                q_chunks = {"time": 1, pl_dim: 13, "y": 1069, "x": 1069}

            if args.drop_other_vars:
                data_subset = data_subset.drop_vars(data_subset.data_vars)
            else:
                data_subset = data_subset.drop_vars(["si10", "wdir10", "r"])

            if args.compute_10m_wind:
                data_subset["10u"] = xr.DataArray(wind_u_values, dims=wind_dims, coords=wind_coords).chunk(wind_chunks)
                data_subset["10v"] = xr.DataArray(wind_v_values, dims=wind_dims, coords=wind_coords).chunk(wind_chunks)

            if args.compute_specific_humidity:
                data_subset["q"] = xr.DataArray(specific_humidity_values, dims=q_dims, coords=q_coords).chunk(q_chunks)

            if args.compute_geopotential_at_surface:
                data_subset["zs"] = xr.DataArray(geopotential_values, dims=wind_dims, coords=wind_coords).chunk(wind_chunks)


            if not os.path.exists(args.dst_dir):
                compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
                encoding = {var: {"compressor": compressor} for var in data_subset.data_vars}

                # It is important to set the encoding for the time variable explicitly, otherwise xarray might select
                # an encoding that is not compatible with the later values, resulting in wrong time values
                # See https://github.com/pydata/xarray/issues/5969 and https://github.com/pydata/xarray/issues/3942
                encoding.update({"time": {"units": "nanoseconds since 1970-01-01"}})

                data_subset.to_zarr(args.dst_dir, mode="w", encoding=encoding)
            else:
                data_subset.to_zarr(args.dst_dir, append_dim="time")

            wind_u_values = []
            wind_v_values = []
            specific_humidity_values = []
            geopotential_values = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        type=str,
        help="Location of the CERRA dataset",
    )

    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Location of the CERRA dataset",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Frequency to save the precomputed variables. Lower values will consume less RAM",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start datetime for the precomputation",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End datetime for the precomputation",
    )

    parser.add_argument(
        "--compute_10m_wind",
        action="store_true",
        help="Whether to compute 10m wind components",
    )

    parser.add_argument(
        "--compute_specific_humidity",
        action="store_true",
        help="Whether to compute specific humidity",
    )

    parser.add_argument(
        "--compute_geopotential_at_surface",
        action="store_true",
        help="Whether to compute geopotential at surface",
    )

    parser.add_argument(
        "--drop_other_vars",
        action="store_true",
        help="If set, drop all other variables except the ones that are computed",
    )

    args = parser.parse_args()
    compute_any = args.compute_10m_wind or args.compute_specific_humidity or args.compute_geopotential_at_surface
    assert compute_any, ("At least one of --compute_10m_wind, --compute_specific_humidity or "
                         "--compute_geopotential_at_surface must be set")

    check_and_start_debugger()
    main(args)
