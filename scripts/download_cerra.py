import argparse
import os
from calendar import monthrange
from typing import Optional

import cdsapi
import xarray as xr
import yaml
from numcodecs import Blosc
from requests.exceptions import ChunkedEncodingError

from main import check_and_start_debugger


def download_chunk(
        config: dict[str, dict],
        outdir: str,
        year: str,
        month: str,
        day: Optional[str] = None,
        ):
    """Download a chunk (either a month or a day) of CERRA data and save it to a zarr file.

        Args:
            config (dict): Template for the CDS API requests
            outdir (str): Directory to save the zarr file to
            year (str): Year to download
            month (str): Month to download
            day (str): Day to download. If None, download the whole month
        """

    tempfiles = []
    is_successful = False

    # The download might fail, so we retry until it succeeds
    while not is_successful:
        is_successful = True
        for i, entry in enumerate(config):
            dataset_config = config[entry]
            dataset = dataset_config["dataset"]
            file_type = dataset_config["request"]["data_format"]
            file_suffix = "nc" if file_type == "netcdf" else "grib"
            request = dataset_config["request"].copy()
            request["year"] = [year]
            request["month"] = [month]

            if day is not None:
                request["day"] = [day]
                tempfile = f"./{entry}_{year}_{month}_{day}.{file_suffix}"
            else:
                tempfile = f"./{entry}_{year}_{month}.{file_suffix}"

            client = cdsapi.Client()
            try:
                client.retrieve(dataset, request, target=tempfile)
            except ChunkedEncodingError as e:
                if day is not None:
                    print(f"Failed to download {entry} for {year}-{month}-{day}: {str(e)}")
                else:
                    print(f"Failed to download {entry} for {year}-{month}: {str(e)}")

                # Clean up
                os.remove(tempfile)
                for tmp in tempfiles:
                    os.remove(tmp)
                is_successful = False
                break

            tempfiles.append(tempfile)

    datasets = []

    for tempfile, entry in zip(tempfiles, config):
        dataset_config = config[entry]
        chunks = dataset_config["chunks"]
        ds = xr.open_dataset(tempfile, chunks=chunks)

        if "reset_coords" in dataset_config:
            ds = ds.reset_coords(dataset_config["reset_coords"])

        if "drop_vars" in dataset_config:
            ds = ds.drop_vars(dataset_config["drop_vars"])

        if "rename_vars" in dataset_config:
            ds = ds.rename(dataset_config["rename_vars"])

        datasets.append(ds)

    ds_merged = xr.merge(datasets)

    if os.path.exists(outdir):
        ds_merged.to_zarr(outdir, append_dim="time", mode="a")
    else:
        # We use lz4hc because it combines good compression with fast decompression
        # It's relatively slow to compress, but we only need to do that once
        compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds_merged.data_vars}

        # It is important to set the encoding for the time variable explicitly, otherwise xarray might select
        # an encoding that is not compatible with the later values, resulting in wrong time values
        # See https://github.com/pydata/xarray/issues/5969 and https://github.com/pydata/xarray/issues/3942
        encoding.update({"time": {"units": "nanoseconds since 1970-01-01"}})

        ds_merged.to_zarr(outdir, mode="w-", encoding=encoding)

    # Clean up
    for tempfile in tempfiles:
        os.remove(tempfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request_template",
        type=str,
        help="Template file for the request",
    )

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
        "--start_day",
        type=int,
        default=1,
        help="Day in start_month to start downloading"
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
        "--by_day",
        default=False,
        action="store_true",
        help="Download data by day instead of by month"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
    )

    check_and_start_debugger()
    args = parser.parse_args()

    with open(args.request_template, "r") as f:
        request_config = yaml.load(f, Loader=yaml.FullLoader)

    current_year = args.start_year
    current_month = args.start_month
    current_day = args.start_day
    end_year = args.end_year
    end_month = args.end_month

    while current_year < end_year or (current_year == end_year and current_month <= end_month):

        if args.by_day:
            num_days_in_month = monthrange(current_year, current_month)[1]
            while current_day <= num_days_in_month:
                download_chunk(
                    config=request_config,
                    outdir=args.outdir,
                    year=f"{current_year}",
                    month=f"{current_month:02}",
                    day=f"{current_day:02}"
                )
                current_day += 1

            current_day = 1
        else:
            download_chunk(
                config=request_config,
                outdir=args.outdir,
                year=f"{current_year}",
                month=f"{current_month:02}"
            )

        print(f"Downloaded CERRA data for {current_year}-{current_month:02}")

        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
