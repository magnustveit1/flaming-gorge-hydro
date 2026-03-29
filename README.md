# Flaming Gorge Reservoir — Water Supply Forecast

**Repository:** https://github.com/magnustveit1/flaming-gorge-hydro

A hydrologic analysis pipeline for the Flaming Gorge Reservoir. Developed for homework #2 at the University of Utah for CVEEN 6920, Hydroinformatics. The pipeline retrieves snow water equivalent (SWE) data from SNOTEL stations, daily streamflow records from USGS NWIS gauges, and watershed boundary data from the USGS NHD, then produces a suite of figures supporting a water supply forecast as of April 1, 2025.

The analysis is structured around four components: (1) a basin map locating the watershed, SNOTEL stations, and stream gauges; (2) historical SWE envelopes for each SNOTEL station with the current water year trace overlaid; (3) historical monthly streamflow volume distributions for April through September with the current water year highlighted; and (4) parity plots relating peak SWE to monthly streamflow volume for each station, with linear regression and 95% prediction intervals. All configurations such as station IDs, dates, and display settings can be toggled and changed in the centralized master script `flaming_gorge_forecast.py`, making the supporting scripts reusable for any reservoir and watershed.

![Flaming Gorge Basin Map](figures/fig1_map.png)

## Repository Structure
```
flaming-gorge-hydro/
├── flaming_gorge_forecast.py   # Master script
├── supporting_scripts/
│   ├── data_acquisition.py     # Fetches basin, SNOTEL, and streamflow data
│   ├── analysis.py             # Computes SWE envelopes, flow volumes, parity stats
│   └── plotting.py             # Produces all figures
├── data/
│   ├── basin/                  # Watershed boundary GeoJSON
│   ├── SNOTEL/                 # SNOTEL SWE CSVs
│   └── NWIS/                   # USGS streamflow CSVs
├── figures/                    # Output figures
├── cashe/                      # Cached data (git-ignored)
├── environment.yml             # Conda environment specification
├── .gitignore
└── LICENSE
```

## Data Sources

- **SNOTEL SWE:** NRCS via [egagli/snotel_ccss_stations](https://github.com/egagli/snotel_ccss_stations) — daily auto-updating CSVs, no API required
- **Streamflow:** USGS NWIS daily values via `dataretrieval`
- **Watershed boundary:** USGS NHD via NLDI (`pynhd`)
- **Basemap:** © OpenTopoMap contributors, © OpenStreetMap contributors

## Usage

**1. Create and activate the environment:**
```bash
conda env create -f environment.yml
conda activate flaming-gorge
```

**2. Run the full pipeline:**
```bash
python flaming_gorge_forecast.py
```

All data is fetched and cached automatically on first run. Subsequent runs load from cache. To re-fetch fresh data, delete the relevant files in `data/`.

## Configuration

All project-specific settings are defined at the top of `flaming_gorge_forecast.py`:

| Variable | Description |
|---|---|
| `INLET_ID` | USGS station ID for reservoir inlet gauge |
| `OUTLET_ID` | USGS station ID for reservoir outlet gauge |
| `ANALYSIS_DATE` | Forecast date (default: April 1, 2025) |
| `START_DATE` | Historical record start date |
| `INLET_NAME` | Human-readable inlet gauge name |
| `OUTLET_NAME` | Human-readable outlet gauge name |
| `RESERVOIR_NAME` | Reservoir name for figure titles |
| `BASIN_NAME` | Basin name for figure titles |
| `MAP_ZOOM` | Basemap tile zoom level |
| `MAP_PADDING_X` | Horizontal map extent padding |
| `MAP_PADDING_Y` | Vertical map extent padding |

## Requirements

See `environment.yml`. Key packages: `geopandas`, `contextily`, `pynhd`, `dataretrieval`, `matplotlib`, `scipy`, `adjusttext`.

## Author

Magnus Tveit - University of Utah, Department of Geography