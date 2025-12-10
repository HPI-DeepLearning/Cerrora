import argparse
from pathlib import Path

import xarray as xr


VARIABLE_LIST = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
    "total_precipitation_6hr",
]

PRESSURE_LEVEL_LIST = [
    50, 100, 150,
    200, 250, 300,
    400, 500, 600,
    700, 850, 925,
    1000
]

PREDICTION_TIMEDELTA_LIST = ["0h", "6h", "12h", "18h", "24h"]

def download_ifs_forecasts(
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        location: str
    ):
    gcp_bucket = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
    res_lon, res_lat = 1440, 721  # Fixed resolution for IFS forecasts
    era5 = xr.open_zarr(
        gcp_bucket,
        chunks={"time": 1, "prediction_timedelta": 1, "longitude": int(res_lon), "latitude": int(res_lat), "level": 13},
        overwrite_encoded_chunks=True
    )
    era5 = era5[VARIABLE_LIST]
    era5 = era5.sel(level=PRESSURE_LEVEL_LIST, prediction_timedelta=PREDICTION_TIMEDELTA_LIST)

    current_year = start_year
    current_month = start_month

    print(f"Downloading ERA5 data from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        # Select correct time range and variables
        era5_month = era5.sel(time=f"{current_year}-{current_month:02d}")

        # Write to disk, appending to the existing dataset if it exists
        if Path(location).exists():
            era5_month.to_zarr(location, append_dim="time", mode="a")
        else:
            era5_month.to_zarr(location, mode="w-")

        if current_month == 12:
            current_month = 1
            current_year += 1

            print(f"Downloaded IFS-HRES data for {current_year-1}")
        else:
            current_month += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_year",
        type=int,
        default=2020,
        help="Year to start downloading"
    )

    parser.add_argument(
        "--start_month",
        type=int,
        default=1,
        help="Month in start_year to start downloading"
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2020,
        help="Year to stop downloading"
    )

    parser.add_argument(
        "--end_month",
        type=int,
        default=12,
        help="Month in end_year to stop downloading"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
    )

    opt = parser.parse_args()

    download_ifs_forecasts(opt.start_year, opt.start_month, opt.end_year, opt.end_month, opt.outdir)
