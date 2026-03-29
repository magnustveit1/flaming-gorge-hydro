"""
data_acquisition.py

Functions to fetch and cache hydrologic data for reservoir water supply
forecasting. Retrieves basin boundaries, SNOTEL snow water equivalent,
and USGS streamflow records. All configuration is passed in as function
arguments.

Data sources:
    - Basin boundary: USGS NHD via NLDI (pynhd)
    - SNOTEL SWE:     egagli/snotel_ccss_stations (GitHub CSV)
    - Streamflow:     USGS NWIS daily values (dataretrieval)

Author: Magnus Tveit
"""

import os
import warnings
import pandas as pd
import geopandas as gpd
from pynhd import NLDI
from dataretrieval import nwis
import sys
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

SNOTEL_BASE_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{}.csv"
SNOTEL_META_URL = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"


def get_basin(inlet_id, data_dir):
    """
    Fetch the NHD watershed boundary for the inlet gauge.

    Parameters
    ----------
    inlet_id : str
        USGS station ID for the reservoir inlet gauge.
    data_dir : str
        Root data directory path.

    Returns
    -------
    basin_gdf : GeoDataFrame
    basin_geom : shapely Polygon
    """
    basin_dir = os.path.join(data_dir, "basin")
    os.makedirs(basin_dir, exist_ok=True)
    path = os.path.join(basin_dir, f"basin_{inlet_id}.geojson")

    if os.path.exists(path):
        print(f"Loading cached basin boundary: {path}")
        basin_gdf = gpd.read_file(path)
    else:
        print(f"Fetching basin boundary for {inlet_id} from NLDI...")
        basin_gdf = NLDI().get_basins(inlet_id)
        basin_gdf.to_file(path, driver="GeoJSON")
        print(f"Saved: {path}")

    return basin_gdf, basin_gdf.geometry.iloc[0]


def get_snotel_stations(basin_geom):
    """
    Filter all SNOTEL stations to those within the basin boundary.

    Parameters
    ----------
    basin_geom : shapely Polygon
        Watershed boundary geometry.

    Returns
    -------
    GeoDataFrame of stations within the basin.
    """
    print("Loading SNOTEL station metadata...")
    stations = gpd.read_file(SNOTEL_META_URL).set_index("code")
    stations = stations[stations["csvData"] == True]
    in_basin = stations[stations.geometry.within(basin_geom)].copy()
    print(f"Found {len(in_basin)} SNOTEL stations in basin:")
    for code, row in in_basin.iterrows():
        print(f"  {code}: {row['name']} ({row['elevation_m']:.0f} m)")
    return in_basin


def get_snotel_data(stations_gdf, data_dir, start_date, end_date):
    """
    Fetch SWE data for each station in stations_gdf.

    Parameters
    ----------
    stations_gdf : GeoDataFrame
        Station metadata, indexed by station code.
    data_dir : str
        Root data directory path.
    start_date : str
        Start date string 'YYYY-MM-DD'.
    end_date : str
        End date string 'YYYY-MM-DD'.

    Returns
    -------
    dict : {station_code: DataFrame}
        Each DataFrame has DatetimeIndex and column WTEQ_m (SWE in meters).
    """
    snotel_dir = os.path.join(data_dir, "SNOTEL")
    os.makedirs(snotel_dir, exist_ok=True)
    snotel_data = {}

    for code in stations_gdf.index:
        path = os.path.join(snotel_dir, f"{code}.csv")

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

        # Clip to analysis period and rename SWE column
        df = df[(df.index >= start_date) & (df.index <= end_date)][["WTEQ"]].copy()
        df.rename(columns={"WTEQ": "WTEQ_m"}, inplace=True)
        df["WTEQ_m"] = pd.to_numeric(df["WTEQ_m"], errors="coerce")
        snotel_data[code] = df
        print(f"  {code}: {len(df)} records, "
              f"{df.index.min().date()} to {df.index.max().date()}")

    return snotel_data


def get_streamflow(station_id, label, data_dir, start_date, end_date):
    """
    Fetch daily mean streamflow from USGS NWIS.

    Parameters
    ----------
    station_id : str
        USGS station ID.
    label : str
        Human-readable label e.g. 'inlet' or 'outlet'.
    data_dir : str
        Root data directory path.
    start_date : str
        Start date string 'YYYY-MM-DD'.
    end_date : str
        End date string 'YYYY-MM-DD'.

    Returns
    -------
    DataFrame with columns [flow_cms, site_no], DatetimeIndex.
    """
    nwis_dir = os.path.join(data_dir, "NWIS")
    os.makedirs(nwis_dir, exist_ok=True)
    path = os.path.join(nwis_dir, f"streamflow_{station_id}.csv")

    if os.path.exists(path):
        print(f"Loading cached streamflow ({label}): {path}")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        print(f"Fetching streamflow ({label}) for {station_id} from NWIS...")
        raw, _ = nwis.get_dv(
            sites=station_id,
            start=start_date,
            end=end_date,
            parameterCd="00060"
        )
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "Date"

        if "00060_Mean" in raw.columns:
            raw.rename(columns={"00060_Mean": "flow_cfs"}, inplace=True)
        elif "00060_00003" in raw.columns:
            raw.rename(columns={"00060_00003": "flow_cfs"}, inplace=True)

        raw["flow_cms"] = raw["flow_cfs"] * 0.0283168
        raw["site_no"]  = station_id
        df = raw[["flow_cms", "site_no"]].copy()
        df.to_csv(path)
        print(f"Saved: {path}")

    print(f"  {station_id} ({label}): {len(df)} records, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


def acquire_all(inlet_id, outlet_id, data_dir, start_date, end_date):
    """
    Run the full data acquisition pipeline.

    Parameters
    ----------
    inlet_id : str
        USGS station ID for reservoir inlet.
    outlet_id : str
        USGS station ID for reservoir outlet.
    data_dir : str
        Root data directory path.
    start_date : str
        Historical record start date 'YYYY-MM-DD'.
    end_date : str
        Analysis cutoff date 'YYYY-MM-DD'.

    Returns
    -------
    dict with keys: basin_gdf, basin_geom, stations_gdf,
                    snotel_data, inlet_df, outlet_df
    """
    print("\n=== Basin Boundary ===")
    basin_gdf, basin_geom = get_basin(inlet_id, data_dir)

    print("\n=== SNOTEL Stations ===")
    stations_gdf = get_snotel_stations(basin_geom)

    print("\n=== SNOTEL Data ===")
    snotel_data = get_snotel_data(stations_gdf, data_dir, start_date, end_date)

    print("\n=== Streamflow: Inlet ===")
    inlet_df = get_streamflow(inlet_id, "inlet", data_dir, start_date, end_date)

    print("\n=== Streamflow: Outlet ===")
    outlet_df = get_streamflow(outlet_id, "outlet", data_dir, start_date, end_date)

    print("\n=== Data acquisition complete ===")
    return {
        "basin_gdf":    basin_gdf,
        "basin_geom":   basin_geom,
        "stations_gdf": stations_gdf,
        "snotel_data":  snotel_data,
        "inlet_df":     inlet_df,
        "outlet_df":    outlet_df,
    }