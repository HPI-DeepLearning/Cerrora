import argparse

import numpy as np
import xarray as xr

from aurora.data.cerra import BOUNDARY_STATIC_VAR_MAPPING
from main import check_and_start_debugger
from scripts.rollout.cerra_region_cropper import CerraRegionCropper
from huggingface_hub import hf_hub_download


def add_static_vars(
        output_path: str,
        lon_lat_path: str,
        boundary_size: int,
        num_cores: int,
):
    """
    Project the static variables from the Aurora dataset onto the CERRA domain
    and add them to the existing regridded dataset.

    Args:
        output_path (str): Path to store the regridded dataset.
        lon_lat_path (str): Path to the CERRA domain's longitude and latitude coordinates.
        boundary_size (int): Size of the boundary to create around the CERRA domain in pixels.
        num_cores (int): Number of cores to use for interpolation. Default is 1, which means no parallel processing.
    """
    static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.1-static.nc")
    static_dataset = xr.open_dataset(static_path)
    output_dataset = xr.open_zarr(output_path)

    boundary_sizes = (boundary_size, boundary_size, boundary_size, boundary_size)
    cropper = CerraRegionCropper(lon_lat_path, boundary_width=boundary_sizes, num_cores=num_cores)

    num_longitudes = static_dataset.longitude.size
    num_latitudes = static_dataset.latitude.size

    # The actual lat/lon values in aurora-0.1-static.nc are not correct,
    # so we load the correct ones from separate files
    lat = np.load("./aurora-0.1-lat.npy")
    lon = np.load("./aurora-0.1-lon.npy")
    lat_matrix = np.repeat(lat[:, np.newaxis], num_longitudes, axis=1)
    lon_matrix = np.repeat(lon[np.newaxis, :], num_latitudes, axis=0)

    data_vars = {}

    for aurora_key, dataset_key in BOUNDARY_STATIC_VAR_MAPPING.items():
        var_values = static_dataset[aurora_key].values
        data_vars[dataset_key] = (["y", "x"], var_values)

    boundary_dataset = xr.Dataset(
        coords={
            "time": output_dataset.time.values,
            "prediction_timedelta": output_dataset.prediction_timedelta.values,
            "pressure_level": output_dataset.pressure_level.values,
            "latitude": (("y", "x"), lat_matrix),
            "longitude": (("y", "x"), lon_matrix),
        },
        data_vars=data_vars,
    )

    boundary_dataset = cropper.crop_and_interpolate(boundary_dataset)
    boundary_dataset.to_zarr(output_path, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    add_static_vars(
        output_path=args.output_path,
        lon_lat_path=args.lon_lat_path,
        boundary_size=args.boundary_size,
        num_cores=args.num_cores,
    )
