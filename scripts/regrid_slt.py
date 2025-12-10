"""
The script is used to regrid the source SLT mask taken from ERA5 to the Lambert Conformal Conic Grid used by the
CERRA Dataset. The re-gridded SLT mask is then appended to the existing CERRA Dataset.
"""

import argparse
import xarray as xr
import numpy as np
import xesmf as xe


def regrid_data(source_ds: xr.DataArray, target_ds: xr.Dataset) -> np.ndarray:
    """
    Takes the source dataset (ERA5 SLT mask), applies coordinate transformation to match the CERRA coordinate range,
    and re-grids the data to match the projection used in CERRA (Lambert Conformal Conic). New data added with
    bilinear interpolation.

    Args:
        source_ds (xr.DataArray): Input dataArray - SLT datavar taken from ERA5.
        target_ds (xr.Dataset): Target dataset - CERRA.

    Returns:
        regridded_data_expanded (np.ndarray): numpy array containing re-gridded data
    """

    # convert longitude to match cerra format
    source_ds = source_ds.assign_coords(
        longitude=((source_ds.longitude + 180) % 360) - 180
    )
    # Sort longitudes for proper ordering
    source_ds = source_ds.sortby("longitude")

    source_lat = source_ds.latitude.values
    source_lon = source_ds.longitude.values

    # xr.DataArray does not need a datavar argument
    slt_var = source_ds.values

    target_lat = target_ds.latitude.values
    target_lon = target_ds.longitude.values

    # Define source and target grids
    source_grid = {
        'lat': source_lat,
        'lon': source_lon,
    }
    target_grid = {
        'lat': target_lat,
        'lon': target_lon,
    }

    # Create the regridder
    regridder = xe.Regridder(source_grid, target_grid, method='nearest_s2d')
    regridded_data = regridder(slt_var)  # Shape will match cerra_lat and cerra_lon

    return regridded_data


def store_zarr(input_data: np.ndarray, target_ds: xr.Dataset, target_path: str) -> None:
    """
    Transforms the re-gridded data to a xr.DataArray, and appends the data to the pre-existing CERRA dataset.
    The resulting DataVariable will have the same shape as other static variables in CERRA (valid_time, y, x)

    Args:
        input_data (np.ndarray): Regridded soil type data.
        target_ds (xr.Dataset): Target dataset used for grid reference and coordinates.
        target_path (str): Path to the target Zarr dataset.
    """

    data = xr.DataArray(
        input_data,
        dims=("y", "x"),
        coords={
            "latitude": (("y", "x"), target_ds.latitude.values),  # Use existing latitude
            "longitude": (("y", "x"), target_ds.longitude.values),  # Use existing longitude
        },
        name="slt",
        attrs={
            "long_name": "Soil Land Type",
            "standard_name": "soil_land_type",
            "units": "dimensionless",
            "description": "Soil type data regridded to match the CERRA grid.",
        }
    )

    data = data.chunk((1069, 1069))  # Apply the chunk size
    print(f'Appending data to CERRA dataset.')
    data.to_zarr(
            target_path,
            mode="a",
            safe_chunks=False # weird bug coming from xarray, throws an error without this arg
    )


def main():

    parser = argparse.ArgumentParser(description="Regrid the ERA5 Soil Land Type mask to match the Cerra Grid.")
    parser.add_argument(
        "--cerra_zarr_path",
        required=True,
        help="Path to the target CERRA Zarr dataset.",
        )
    parser.add_argument(
        "--slt_zarr_path",
        required=False,
        help="Path to the GCP bucket containing SLT mask.",
        default='gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
    )
    args = parser.parse_args()

    cerra_ds = xr.open_zarr(args.cerra_zarr_path)
    slt = xr.open_zarr(args.slt_zarr_path)['soil_type']

    regridded = regrid_data(slt, cerra_ds)
    store_zarr(regridded, cerra_ds, args.cerra_zarr_path)

if __name__ == "__main__":
    main()
