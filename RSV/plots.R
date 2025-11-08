# read all parquet files in RSV/data

library(arrow)
library(tidyverse)
library(scattermore)

# ------------------------------------
#           RSV SMH data
# ------------------------------------
smh <- arrow::read_parquet("RSV/data/rsv_smh/output/rsv_smh.parquet")
smh |> glimpse()

ggplot(smh, aes(x = week_enddate, y = value, colour = datasetH2)) +
  geom_scattermore(alpha = 0.5) +
  theme_classic() +
  theme(legend.position = "none")

ggplot(smh, aes(x = week_enddate, y = value, colour = datasetH2)) +
  facet_grid(rows = vars(fluseason), scales = "free_y") +
  geom_scattermore(alpha = 0.5) +
  theme_classic() +
  theme(legend.position = "none")


# ------------------------------------
#           Surveillance Data
# ------------------------------------
# --------- RSV-NET ---------
rsvnet <- arrow::read_parquet("RSV/data/rsvnet.parquet")
glimpse(rsvnet)
mean(is.na(rsvnet$value)) # !!!
ggplot(rsvnet, aes(x = fluseason_fraction, y = value, colour = location_code)) +
  facet_grid(rows = vars(fluseason), scales = "free") +
  geom_line() +
  theme_classic() +
  theme(legend.position = "none")

# --------- NHSN ---------
nhsn <- arrow::read_parquet("RSV/data/nhsn.parquet")
glimpse(nhsn)
nhsn$location_code |> table(useNA = "ifany") # investigate!!
nhsn$value |> summary()

ggplot(nhsn, aes(x = fluseason_fraction, y = value, colour = location_code)) +
  facet_grid(rows = vars(fluseason), scales = "free") +
  geom_point() +
  theme_classic() +
  theme(legend.position = "none")

# --------- NSSP ---------
nssp <- arrow::read_parquet("RSV/data/nssp.parquet")
glimpse(nssp)
nssp$location_code |> table(useNA = "ifany") # investigate!!
nssp$value |> summary()
ggplot(nssp, aes(x = fluseason_fraction, y = value, colour = location_code)) +
  facet_grid(rows = vars(fluseason), scales = "free") +
  geom_line() +
  theme_classic() +
  theme(legend.position = "none")
