"""
data_acquisition.py

Fetches and caches all data needed for the Flaming Gorge reservoir
water supply forecast:
  - Basin boundary (NLDI)
  - SNOTEL stations within basin (egagli/snotel_ccss_stations)
  - USGS streamflow: inlet (09217000) and outlet (09234500)

All data is saved to the data/ directory. Functions are
cache-aware: if a file already exists it is loaded, not re-fetched.

Author: Magnus Tveit
"""

import os
import warnings
import pandas as pd
import geopandas as gpd
from pynhd import NLDI
from dataretrieval import nwis
warnings.filterwarnings("ignore")

# Config
INLET_ID    = "09217000"   # Green River near Green River, WY
OUTLET_ID   = "09234500"   # Green River near Greendale, UT (dam outlet)
ANALYSIS_DATE = "2025-04-01"
START_DATE  = "1980-01-01"  # full historical record
END_DATE    = "2025-04-01"  # analysis cutoff

SNOTEL_BASE_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{}.csv"
SNOTEL_META_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"

DATA_DIR        = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SNOTEL_DIR      = os.path.join(DATA_DIR, "SNOTEL")
NWIS_DIR        = os.path.join(DATA_DIR, "NWIS")
BASIN_DIR       = os.path.join(DATA_DIR, "basin")

# Helpers

def _make_dirs():
    """Create all required output directories."""
    for d in [SNOTEL_DIR, NWIS_DIR, BASIN_DIR]:
        os.makedirs(d, exist_ok=True)


# Basin boundary

def get_basin(station_id=INLET_ID):
    """
    Fetch the NHD watershed boundary for the inlet gauge.
    Returns a GeoDataFrame and the shapely geometry.
    Cached to data/basin/basin_{station_id}.geojson.
    """
    path = os.path.join(BASIN_DIR, f"basin_{station_id}.geojson")

    if os.path.exists(path):
        print(f"Loading cached basin boundary: {path}")
        basin_gdf = gpd.read_file(path)
    else:
        print(f"Fetching basin boundary for {station_id} from NLDI...")
        basin_gdf = NLDI().get_basins(station_id)
        basin_gdf.to_file(path, driver="GeoJSON")
        print(f"Saved: {path}")

    basin_geom = basin_gdf.geometry.iloc[0]
    return basin_gdf, basin_geom


# SNOTEL

def get_snotel_stations(basin_geom):
    """
    Load all SNOTEL stations from egagli/snotel_ccss_stations,
    filter to those within the basin boundary.
    Returns a GeoDataFrame of stations.
    """
    print("Loading SNOTEL station metadata...")
    stations = gpd.read_file(SNOTEL_META_URL).set_index("code")
    # keep only stations with actual CSV data
    stations = stations[stations["csvData"] == True]
    stations_in_basin = stations[stations.geometry.within(basin_geom)].copy()
    print(f"Found {len(stations_in_basin)} SNOTEL stations in basin:")
    for code, row in stations_in_basin.iterrows():
        print(f"  {code}: {row['name']} ({row['elevation_m']:.0f} m)")
    return stations_in_basin


def get_snotel_data(stations_gdf):
    """
    Fetch SWE data for each station in stations_gdf.
    Returns a dict: {station_code: DataFrame} with datetime index
    and column WTEQ_m (SWE in meters).
    Cached to data/SNOTEL/{code}.csv.
    """
    snotel_data = {}

    for code in stations_gdf.index:
        path = os.path.join(SNOTEL_DIR, f"{code}.csv")

        if os.path.exists(path):
            print(f"Loading cached SNOTEL: {code}")
            df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        else:
            print(f"Fetching SNOTEL data: {code}")
            url = SNOTEL_BASE_URL.format(code)
            try:
                df = pd.read_csv(url, index_col="datetime", parse_dates=True)
                df.to_csv(path)
                print(f"  Saved: {path}")
            except Exception as e:
                print(f"  WARNING: Could not fetch {code}: {e}")
                continue

        # Clip to analysis period and keep only SWE
        df = df[(df.index >= START_DATE) & (df.index <= END_DATE)][["WTEQ"]].copy()
        # WTEQ is in meters per the egagli repo docs
        df.rename(columns={"WTEQ": "WTEQ_m"}, inplace=True)
        df["WTEQ_m"] = pd.to_numeric(df["WTEQ_m"], errors="coerce")

        snotel_data[code] = df
        print(f"  {code}: {len(df)} records, "
              f"{df.index.min().date()} to {df.index.max().date()}")

    return snotel_data


# USGS Streamflow

def get_streamflow(station_id, label="inlet"):
    """
    Fetch daily mean streamflow from USGS NWIS for a given station.
    Converts from cfs to cms.
    Returns a DataFrame with columns [flow_cms, site_no].
    Cached to data/NWIS/streamflow_{station_id}.csv.
    """
    path = os.path.join(NWIS_DIR, f"streamflow_{station_id}.csv")

    if os.path.exists(path):
        print(f"Loading cached streamflow ({label}): {path}")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        print(f"Fetching streamflow ({label}) for {station_id} from NWIS...")
        raw, _ = nwis.get_dv(
            sites=station_id,
            start=START_DATE,
            end=END_DATE,
            parameterCd="00060"   # discharge in cfs
        )
        # Strip timezone info
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "Date"

        # Rename flow column — NWIS returns '00060_Mean'
        if "00060_Mean" in raw.columns:
            raw.rename(columns={"00060_Mean": "flow_cfs"}, inplace=True)
        elif "00060_00003" in raw.columns:
            raw.rename(columns={"00060_00003": "flow_cfs"}, inplace=True)

        raw["flow_cms"] = raw["flow_cfs"] * 0.0283168
        raw["site_no"] = station_id
        df = raw[["flow_cms", "site_no"]].copy()
        df.to_csv(path)
        print(f"Saved: {path}")

    print(f"  {station_id} ({label}): {len(df)} records, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


# Main

def acquire_all():
    """
    Run the full data acquisition pipeline.
    Returns all fetched data as a dict for use by analysis.py.
    """
    _make_dirs()

    print("\n=== Basin Boundary ===")
    basin_gdf, basin_geom = get_basin()

    print("\n=== SNOTEL Stations ===")
    stations_gdf = get_snotel_stations(basin_geom)

    print("\n=== SNOTEL Data ===")
    snotel_data = get_snotel_data(stations_gdf)

    print("\n=== Streamflow: Inlet ===")
    inlet_df = get_streamflow(INLET_ID, label="inlet")

    print("\n=== Streamflow: Outlet ===")
    outlet_df = get_streamflow(OUTLET_ID, label="outlet")

    print("\n=== Data acquisition complete ===")
    return {
        "basin_gdf":    basin_gdf,
        "basin_geom":   basin_geom,
        "stations_gdf": stations_gdf,
        "snotel_data":  snotel_data,
        "inlet_df":     inlet_df,
        "outlet_df":    outlet_df,
    }


if __name__ == "__main__":
    acquire_all()