import argparse
import os

import numpy as np
import xarray as xr
from numcodecs import Blosc
from tqdm import tqdm

from main import check_and_start_debugger
from scripts.rollout.cerra_region_cropper import CerraRegionCropper


def interpolate_timestep(
        global_dataset: xr.Dataset,
        cropper: CerraRegionCropper,
        start_datetime: np.datetime64,
        end_datetime: np.datetime64,
) -> xr.Dataset:
    num_longitudes = global_dataset.longitude.size
    num_latitudes = global_dataset.latitude.size
    lat_matrix = np.repeat(global_dataset.latitude.values[:, np.newaxis], num_longitudes, axis=1)
    lon_matrix = np.repeat(global_dataset.longitude.values[np.newaxis, :], num_latitudes, axis=0)

    global_dataset_timestep = global_dataset.sel(time=slice(start_datetime, end_datetime + np.timedelta64(1, 's')))

    data_vars = {}

    for var in global_dataset_timestep.data_vars:
        var_values = global_dataset_timestep[var].values
        if var_values.ndim == 5:
            data_vars[var] = (["time", "prediction_timedelta", "pressure_level", "y", "x"], var_values)
        elif var_values.ndim == 4:
            data_vars[var] = (["time", "prediction_timedelta", "y", "x"], var_values)
        else:
            data_vars[var] = (["y", "x"], var_values)

    boundary_dataset = xr.Dataset(
        coords={
            "time": global_dataset_timestep.time.values,
            "prediction_timedelta": global_dataset_timestep.prediction_timedelta.values,
            "pressure_level": global_dataset_timestep.level.values,
            "latitude": (("y", "x"), lat_matrix),
            "longitude": (("y", "x"), lon_matrix),
        },
        data_vars=data_vars,
    )

    boundary_dataset = cropper.crop_and_interpolate(boundary_dataset)

    return boundary_dataset

def create_rollout_boundaries(
        global_path: str,
        start_time: str,
        end_time: str,
        output_path: str,
        lon_lat_path: str,
        boundary_size: int,
        num_cores: int,
):
    """
    Create a boundary around the CERRA domain based on the global forecast dataset.

    Args:
        global_path (str): Path to the global forecast dataset.
        start_time (str): Start time of the period to transform.
        end_time (str): End time of the period to transform.
        output_path (str): Path to store the regridded dataset.
        lon_lat_path (str): Path to the CERRA domain's longitude and latitude coordinates.
        boundary_size (int): Size of the boundary to create around the CERRA domain in pixels.
        num_cores (int): Number of cores to use for interpolation. Default is 1, which means no parallel processing.
    """
    global_dataset = xr.open_zarr(global_path)
    global_dataset = global_dataset.sel(time=slice(start_time, end_time))
    boundary_sizes = (boundary_size, boundary_size, boundary_size, boundary_size)
    cropper = CerraRegionCropper(lon_lat_path, boundary_width=boundary_sizes, num_cores=num_cores)

    # The number of timesteps per interpolate call
    # As the interpolation can be parallelized up to num_timesteps x num_timedeltas times
    # this value is essential to allowing good utilization of higher core counts
    timesteps_per_interpolate = 5
    time_count = 0
    start_datetime = None

    for datetime in tqdm(global_dataset.time.values):
        if time_count == 0:
            start_datetime = datetime

        time_count += 1

        if time_count % timesteps_per_interpolate != 0 and datetime != global_dataset.time.values[-1]:
            # If we are not at the end of the dataset, we continue to the next iteration
            continue
        else:
            time_count = 0
            end_datetime = datetime

        boundary_dataset = interpolate_timestep(global_dataset, cropper, start_datetime, end_datetime)

        if os.path.exists(output_path):
            boundary_dataset.to_zarr(output_path, append_dim="time", mode="a")
        else:
            # We use lz4hc because it combines good compression with fast decompression
            # It's relatively slow to compress, but we only need to do that once
            compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
            encoding = {var: {"compressor": compressor} for var in boundary_dataset.data_vars}
            # It is important to set the encoding for the time variable explicitly, otherwise xarray might select
            # an encoding that is not compatible with the later values, resulting in wrong time values
            # See https://github.com/pydata/xarray/issues/5969 and https://github.com/pydata/xarray/issues/3942
            encoding.update({"time": {"units": "nanoseconds since 1970-01-01"}})
            boundary_dataset.to_zarr(output_path, mode="w-", encoding=encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global_path",
        type=str,
        help="Path to the global forecast dataset. Must be in 0.25 degree resolution.",
    )

    parser.add_argument(
        "--start_time",
        type=str,
        default="2016-01-01T00:00:00",
        help="Start time of the time interval.",
    )

    parser.add_argument(
        "--end_time",
        type=str,
        default="2021-12-31T21:00:00",
        help="End time of the time interval.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to store the regridded dataset"
    )

    parser.add_argument(
        "--lon_lat_path",
        type=str,
        default="./cerra_lon_lat.npz",
        help="Path to the CERRA domain's longitude and latitude coordinates. "
             "This is used to crop the global dataset to the CERRA region."
    )

    parser.add_argument(
        "--boundary_size",
        type=int,
        help="Size of the boundary to create around the CERRA domain in pixels."
    )

    parser.add_argument(
        "--num_cores",
        type=int,
        default=1,
        help="Number of cores to use for interpolation. Default is 1, which means no parallel processing."
    )

    check_and_start_debugger()
    args = parser.parse_args()

    create_rollout_boundaries(
        global_path=args.global_path,
        start_time=args.start_time,
        end_time=args.end_time,
        output_path=args.output_path,
        lon_lat_path=args.lon_lat_path,
        boundary_size=args.boundary_size,
        num_cores=args.num_cores,
    )
