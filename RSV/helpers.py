import us
import pandas as pd
import polars as pl
import datetime
import epiweeks
from pathlib import Path
import requests
import influpaint.utils.season_axis as utils


def date_to_epiweek(d):
    d = datetime.datetime.strptime(d, "%Y-%m-%d").date()
    return epiweeks.Week.fromdate(d)


#  Function to convert epiweek string to week end date
def epiweek_to_enddate(w):
    w = int(w)
    year = w // 100
    week = w % 100
    return epiweeks.Week(year, week).enddate()


def get_season(epiweek_str, cutoff_week=30):
    w = int(epiweek_str)
    year = w // 100
    week = w % 100

    if week < cutoff_week:
        # Early weeks belong to previous season
        season_start = year - 1
        season_end = year
    else:
        season_start = year
        season_end = year + 1

    return f"{season_start}-{season_end}"


def map_state(s):
    if s in ["United States", "USA", "RSV-NET"]:
        return "US"
    state_obj = us.states.lookup(s)
    if state_obj is None:
        return None
    return state_obj.abbr


def state_fips(state_abbr):
    abbr = state_abbr.upper()
    if abbr == "DC":
        return "11"
    if abbr in ["US", "USA"]:
        return "US"

    state = us.states.lookup(abbr)
    if state is None:
        return None
    return str(state.fips)


from influpaint.utils.season_axis import SeasonAxis
import datetime

# Create a reusable SeasonAxis instance
_season_axis = SeasonAxis.for_flusight()


def week_enddate(date):
    """
    Convert a date to the corresponding week-ending Saturday
    in the flu season calendar.

    Parameters
    ----------
    date : str, datetime.date, or datetime.datetime
        Input date (e.g., '2023-09-30').

    Returns
    -------
    datetime.date
        Saturday corresponding to the season week of the input date.
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime.datetime):
        date = date.date()

    season_year = _season_axis.get_fluseason_year(date)
    week = _season_axis.get_season_week(date)
    return _season_axis.week_to_saturday(season_year, week)


def download_github_folder(api_url, dest):
    """Recursively download a folder from a public GitHub repo using the GitHub API."""
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()

    for item in items:
        if item["type"] == "dir":
            subfolder = dest / item["name"]
            subfolder.mkdir(exist_ok=True)
            download_github_folder(item["url"], subfolder)
        elif item["type"] == "file":
            file_path = dest / item["name"]
            print(f"Downloading {file_path}")
            file_data = requests.get(item["download_url"])
            file_data.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(file_data.content)


def process_smh(f):
    df = pl.read_parquet(f)

    return (
        df.filter(
            (pl.col("age_group") == "0-130")
            & (pl.col("target") == "inc hosp")
            & (pl.col("output_type") == "sample")
        )
        .with_columns(pl.col("origin_date").cast(pl.Date).alias("origin_date"))
        .with_columns(
            (
                pl.col("origin_date")
                + pl.duration(days=(pl.col("horizon") - 1) * 7 + 6)
            ).alias("week_enddate")
        )
        .with_columns(
            [
                pl.col("week_enddate")
                .map_elements(
                    lambda d: utils.get_season_week(d, start_month=8, start_day=1),
                    return_dtype=pl.Int64,
                )
                .alias("season_week"),
                pl.col("week_enddate")
                .map_elements(
                    lambda d: utils.get_season_fraction(d, start_month=8, start_day=1),
                    return_dtype=pl.Float64,
                )
                .alias("fluseason_fraction"),
                pl.col("week_enddate")
                .map_elements(
                    lambda d: utils.get_season_year(d, start_month=8, start_day=1),
                    return_dtype=pl.Int64,
                )
                .alias("fluseason"),
                pl.col("location").cast(pl.Categorical).alias("location_code"),
                pl.col("value").cast(pl.Float64),
                pl.concat_str(
                    [pl.col("run_grouping"), pl.col("stochastic_run")], separator="."
                ).alias("sample"),
                pl.lit("rsv_smh").alias("datasetH1"),
                pl.lit(f.name.removesuffix(".gz.parquet")).alias("datasetH2"),
            ]
        )
        .select(
            [
                "week_enddate",
                "location_code",
                "value",
                "fluseason_fraction",
                "season_week",
                "fluseason",
                "datasetH1",
                "datasetH2",
                "sample",
            ]
        )
    )
