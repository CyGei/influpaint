# ------------------------------------
#           Setup
# ------------------------------------
import sys
import os
from pathlib import Path

rsv_path = Path("/Users/cy/Documents/PROJECTS LOCAL/influpaint/RSV")
sys.path.append(str(rsv_path))
root_path = rsv_path.parent
sys.path.append(str(root_path))

import helpers as hp
import influpaint.utils.season_axis as utils

import epidatpy
from sodapy import Socrata
import datetime
import pandas as pd
import polars as pl
import epiweeks
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import us
import requests
import importlib

importlib.reload(hp)

# ------------------------------------
#           Surveillancde Data
# ------------------------------------
# --------- NSSP ---------
# Source: https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/nssp.html
# pct_ed_visits_rsv : Percent of ED visits that had a discharge diagnosis code of rsv
# Earliest date available: 2022-10-01
epidata = epidatpy.EpiDataContext()
nssp = (
    epidata.pub_covidcast(
        data_source="nssp",
        signals="pct_ed_visits_rsv",
        geo_type="state",
        time_type="week",
        time_values=epidatpy.EpiRange(
            hp.date_to_epiweek("2022-10-01"), hp.date_to_epiweek("2025-10-01")
        ),
    )
    .df()
    .assign(
        week_enddate=lambda df: pd.to_datetime(
            df["time_value"].map(hp.epiweek_to_enddate)
        ),
        location_code=lambda df: df["geo_value"].map(hp.state_fips).astype("category"),
        datasetH1="nssp",
        datasetH2="nssp",
        sample=1,
        fluseason=lambda df: df["week_enddate"].map(
            lambda d: utils.get_season_year(d, start_month=8, start_day=1)
        ),
        season_week=lambda df: df["week_enddate"].map(
            lambda d: utils.get_season_week(d, start_month=8, start_day=1)
        ),
        fluseason_fraction=lambda df: df["week_enddate"].map(
            lambda d: utils.get_season_fraction(d, start_month=8, start_day=1)
        ),
        value=lambda df: df["value"].astype(float),
    )
    .loc[
        :,
        [
            "datasetH1",
            "datasetH2",
            "sample",
            "fluseason",
            "location_code",
            "week_enddate",
            "season_week",
            "fluseason_fraction",
            "value",
        ],
    ]
    .sort_values(["location_code", "week_enddate"])
    .reset_index(drop=True)
)
nssp.to_parquet("RSV/data/nssp.parquet")
sns.lineplot(
    data=nssp,
    x="week_enddate",
    y="value",
    hue="location_code",
    legend=False,
    linewidth=0.8,
)

# --------- RSV-NET ---------
# Source: https://data.cdc.gov/Public-Health-Surveillance/Weekly-Rates-of-Laboratory-Confirmed-RSV-Hospitali/29hc-w46k/about_data
# 199K rows
# rate: Weekly Laboratory-confirmed RSV-associated hospitalization rates
rsvnet = (
    pd.read_csv(
        "https://raw.githubusercontent.com/midas-network/rsv-scenario-modeling-hub/main/target-data/time-series.csv"
    )
    .query("age_group == '0-130'")
    .assign(
        date=lambda df: pd.to_datetime(df["date"]),
        fluseason=lambda df: df["date"].map(
            lambda d: utils.get_season_year(d, start_month=8, start_day=1)
        ),
        fluseason_fraction=lambda df: df["date"].map(
            lambda d: utils.get_season_fraction(d, start_month=8, start_day=1)
        ),
        season_week=lambda df: df["date"].map(
            lambda d: utils.get_season_week(d, start_month=8, start_day=1)
        ),
        week_enddate=lambda df: df["date"].map(lambda d: hp.week_enddate(d)),
        location_code=lambda x: x["location"],
        datasetH1="rsvnet",
        datasetH2="rsvnet",
        sample=1,
        value=lambda df: df["observation"],
    )
    .loc[
        :,
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
        ],
    ]
    .sort_values(["location_code", "week_enddate"])
    .reset_index(drop=True)
)
rsvnet.info()
rsvnet.to_parquet("RSV/data/rsvnet.parquet")
sns.lineplot(
    data=rsvnet,
    x="week_enddate",
    y="value",
    hue="location_code",
    legend=False,
    linewidth=0.8,
)


# --------- NHSN -----------
# Source: https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
# 18K rows
# totalconfrsvnewadm: Total number of new hospital admissions of patients with confirmed RSV captured during the reporting week (Sunday - Saturday)
client = Socrata("data.cdc.gov", None)
nhsn = (
    pd.DataFrame.from_records(client.get("ua7e-t2fy", limit=20000))
    .assign(
        week_enddate=lambda df: pd.to_datetime(df["weekendingdate"]),
        fluseason=lambda df: df["week_enddate"].map(
            lambda d: utils.get_season_year(d, start_month=8, start_day=1)
        ),
        fluseason_fraction=lambda df: df["week_enddate"].map(
            lambda d: utils.get_season_fraction(d, start_month=8, start_day=1)
        ),
        season_week=lambda df: df["week_enddate"].dt.isocalendar().week,
        state=lambda df: df["jurisdiction"].map(hp.map_state).astype("category"),
        location_code=lambda df: df["jurisdiction"]
        .map(hp.state_fips)
        .astype("category"),
        value=lambda df: pd.to_numeric(df["totalconfrsvnewadm"], errors="coerce"),
        datasetH1="nhsn",
        datasetH2="nhsn",
        sample=1,
    )
    .loc[
        :,
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
        ],
    ]
    .sort_values(["location_code", "week_enddate"])
    .reset_index(drop=True)
)
nhsn.to_parquet("RSV/data/nhsn.parquet")
sns.lineplot(
    data=nhsn,
    x="week_enddate",
    y="value",
    hue="location_code",
    legend=False,
    linewidth=0.8,
)
nhsn.describe()


# ------------------------------------
#          Modelling Data
# ------------------------------------
# --------- MIDAS - scenario modelling hub ---------
# Source: https://github.com/midas-network/rsv-scenario-modeling-hub
# Contact: Lucie Contamin

out_dir = Path("RSV/data/rsv_smh/input")
out_dir.mkdir(parents=True, exist_ok=True)
hp.download_github_folder(
    "https://api.github.com/repos/midas-network/rsv-scenario-modeling-hub/contents/model-output",
    out_dir,
)

files = [f for f in out_dir.rglob("*.parquet") if "ensemble" not in f.name.lower()]
len(files)
processed = [hp.process_smh(f) for f in files]
processed = pl.concat(processed)

out_dir = Path("RSV/data")
out_dir.mkdir(parents=True, exist_ok=True)
processed.write_parquet(out_dir / "rsv_smh.parquet")


# ------------------------------------
#           Combine Datasets
# ------------------------------------
# read all parquet in RSV/data (no subfolders)
data_path = Path("RSV/data")
parquet_files = [f for f in data_path.rglob("*.parquet") if f.parent == data_path]
# exclude rsv_smh.parquet
parquet_files = [f for f in parquet_files if f.name != "rsv_smh.parquet"]

surv = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
smh = pl.read_parquet(data_path / "rsv_smh.parquet").to_pandas()

combined = pd.concat([surv, smh], ignore_index=True)
combined["week_enddate"] = pd.to_datetime(combined["week_enddate"], errors="coerce")
combined["sample"] = combined["sample"].astype(str)
combined.info()
combined.describe()
combined
combined.to_parquet("RSV/data/rsv_combined.parquet")
