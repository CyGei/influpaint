# Dataset Structure Description

The dataset `all_datasets.parquet` contains weekly surveillance or model-based estimates of respiratory virus activity, structured by season, location, and data source. Each row represents a single observation defined by a combination of dataset identifiers, location, and reporting week.

## Columns

| Column | Type | Description | function |
|:-------|:-----|:------------|:---------|
| `week_enddate` | datetime | Week-ending date corresponding to the reporting period. | `epiweeks.enddate()` |
| `location_code` | character | Location identifier, expressed as a FIPS code. | `us.states.lookup()`. Note: "DC"=`11` and "US/USA"=`US` |
| `value` | numeric | Reported or modelled incidence for the given `week_enddate` and `location_code`. Rates from surveillance data must be converted to counts using the population size of the corresponding `location_code`. |  |
| `fluseason_fraction` | numeric | Fractional position of the week within the influenza season, ranging from 0 (start) to 1 (end), see influpaint utilities. | see `get_season_fraction()` in `influpaint/utils/season_axis.py` |
| `season_week` | integer | Season week number (like epiweek but for flu seasons), see influpaint utilities. | see `get_season_week()` in `influpaint/utils/season_axis.py` |
| `fluseason` | integer | Influenza season year, see influpaint utilities. | see `get_fluseason_year()` in `influpaint/utils/season_axis.py` |
| `datasetH1` | character | Identifier of the **main dataset** (e.g. RSV-Net or SMHR4). |  |
| `datasetH2` | character | Identifier of the **subdataset**, when applicable. If not applicable, identical to datasetH1. |  |
| `sample` | character | Sample label distinguishing multiple samples within the same dataset and season. Always 1 for surveillance data. |  |


# Questions for Joseph/Shaun:
- RSV SMH: how are we treating `run_grouping` and `stochastic_run`? I pasted them as <`run_grouping`>.<`stochastic_run`> in the `sample` column for RSV SMH data.