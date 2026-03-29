"""
analysis.py

Computes all statistics needed for the Flaming Gorge water supply forecast:

  Section 2 - SWE Analysis:
    - Historical SWE envelope (min, max, median, 25th/75th pct) by day-of-year
    - April 1 2025 SWE vs historical median per station
    - Peak SWE per water year per station

  Section 3 - Streamflow Analysis:
    - Monthly volumetric flow (m³) for Apr-Sep per water year
    - Historical range (min/max/median/25th/75th pct) per month

  Section 4 - Peak SWE vs Monthly Streamflow:
    - Peak SWE per water year paired with Apr-Sep monthly volume
    - Ready for 6 parity/scatter plots

All functions accept the data dict returned by data_acquisition.acquire_all()
and return clean DataFrames. No plotting here — that lives in plotting.py.

Author: Magnus Tveit
"""

import warnings
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# Config
ANALYSIS_DATE   = pd.Timestamp("2025-04-01")
FORECAST_MONTHS = [4, 5, 6, 7, 8, 9]   # Apr–Sep
MONTH_NAMES     = {4:"April", 5:"May", 6:"June",
                   7:"July",  8:"August", 9:"September"}

# Minimum number of years required to include a station/month in statistics
MIN_YEARS = 10


# Helpers

def _water_year(dt):
    """Return the water year for a datetime (Oct 1 starts a new WY)."""
    return dt.year + 1 if dt.month >= 10 else dt.year


def _add_water_year(df):
    """Add a 'water_year' column to a DataFrame with a DatetimeIndex."""
    df = df.copy()
    df["water_year"] = df.index.map(_water_year)
    return df


def _doy_label(dt):
    """
    Return a day-of-water-year label (Oct 1 = day 1).
    Used to align years for envelope plots.
    """
    wy_start = pd.Timestamp(f"{_water_year(dt) - 1}-10-01")
    return (dt - wy_start).days + 1


# SWE Analysis

def compute_swe_envelope(snotel_df, station_code):
    """
    Compute historical SWE envelope for a single station.

    Parameters
    ----------
    snotel_df : DataFrame
        Raw station data with DatetimeIndex and column WTEQ_m.
    station_code : str
        Station identifier, used for labeling.

    Returns
    -------
    envelope : DataFrame
        Indexed by day-of-water-year (1-366).
        Columns: median, mean, q25, q75, min, max.
        Only computed from years with >= MIN_YEARS of data.
    None if insufficient data.
    """
    df = snotel_df[["WTEQ_m"]].copy()
    df = _add_water_year(df)

    # Exclude the current water year (WY2025) from historical stats
    # so the forecast year isn't baked into the baseline
    historical = df[df["water_year"] < 2025].copy()

    # Add day-of-water-year
    historical["dowy"] = historical.index.map(_doy_label)

    # Only keep Oct–Jun (days 1–275 approx) — SWE is near zero Jul–Sep
    historical = historical[historical["dowy"] <= 275]

    # Check we have enough years
    n_years = historical["water_year"].nunique()
    if n_years < MIN_YEARS:
        print(f"  WARNING: {station_code} only has {n_years} historical years "
              f"(need {MIN_YEARS}) — skipping envelope")
        return None

    # Group by day-of-water-year and compute stats
    # min_count=1 ensures we don't compute stats from all-NaN days
    grouped = historical.groupby("dowy")["WTEQ_m"]
    envelope = pd.DataFrame({
        "median": grouped.median(),
        "mean":   grouped.mean(),
        "q25":    grouped.quantile(0.25),
        "q75":    grouped.quantile(0.75),
        "min":    grouped.min(),
        "max":    grouped.max(),
        "n_years": grouped.count(),
    })

    return envelope


def compute_peak_swe_per_year(snotel_df, station_code):
    """
    Compute peak (maximum) SWE for each water year at a station.

    Returns
    -------
    DataFrame with columns [water_year, peak_swe_m].
    Water years with all-NaN SWE are dropped.
    """
    df = snotel_df[["WTEQ_m"]].copy()
    df = _add_water_year(df)

    # Only look at Oct–Jun for peak (Jul–Sep SWE is trivially 0)
    df["month"] = df.index.month
    df = df[df["month"].isin([10,11,12,1,2,3,4,5,6])]

    peak = (df.groupby("water_year")["WTEQ_m"]
              .max()
              .dropna()
              .reset_index()
              .rename(columns={"WTEQ_m": "peak_swe_m"}))

    return peak


def compute_april1_swe(snotel_df, stations_gdf):
    """
    Extract April 1 2025 SWE for each station and compare to
    historical April 1 median.

    Returns
    -------
    DataFrame with one row per station:
        code, name, elevation_m, swe_2025_m, hist_median_m,
        pct_of_median, swe_2025_in, hist_median_in
    """
    records = []

    for code, station_df in snotel_df.items():
        df = station_df[["WTEQ_m"]].copy()

        # April 1 2025 value
        try:
            val_2025 = df.loc[ANALYSIS_DATE, "WTEQ_m"]
        except KeyError:
            # Try nearest date within 3 days
            nearby = df.loc["2025-03-29":"2025-04-03", "WTEQ_m"].dropna()
            val_2025 = nearby.iloc[-1] if len(nearby) > 0 else np.nan

        # Historical April 1 values (exclude 2025)
        april1_hist = df[
            (df.index.month == 4) &
            (df.index.day == 1) &
            (df.index.year < 2025)
        ]["WTEQ_m"].dropna()

        hist_median = april1_hist.median() if len(april1_hist) >= 5 else np.nan
        pct_of_median = (val_2025 / hist_median * 100
                         if hist_median and hist_median > 0 else np.nan)

        name = stations_gdf.loc[code, "name"] if code in stations_gdf.index else code
        elev = stations_gdf.loc[code, "elevation_m"] if code in stations_gdf.index else np.nan

        records.append({
            "code":           code,
            "name":           name,
            "elevation_m":    elev,
            "swe_2025_m":     val_2025,
            "hist_median_m":  hist_median,
            "pct_of_median":  pct_of_median,
            # convert to inches for intuition (standard SNOTEL reporting unit)
            "swe_2025_in":    val_2025 * 39.3701 if not np.isnan(val_2025) else np.nan,
            "hist_median_in": hist_median * 39.3701 if not np.isnan(hist_median) else np.nan,
        })

    return pd.DataFrame(records).set_index("code")


def run_swe_analysis(data):
    """
    Run all SWE analyses. Returns dict with keys:
        envelopes   : {station_code: envelope_df}
        peak_swe    : {station_code: peak_df}
        april1_table: DataFrame (one row per station)
    """
    snotel_data   = data["snotel_data"]
    stations_gdf  = data["stations_gdf"]

    print("\n--- SWE Analysis ---")

    envelopes = {}
    peak_swe  = {}

    for code, df in snotel_data.items():
        print(f"  Processing {code}...")
        env = compute_swe_envelope(df, code)
        if env is not None:
            envelopes[code] = env
        pk = compute_peak_swe_per_year(df, code)
        if len(pk) >= MIN_YEARS:
            peak_swe[code] = pk

    april1_table = compute_april1_swe(snotel_data, stations_gdf)

    print("\nApril 1 2025 SWE summary:")
    print(april1_table[["name","swe_2025_in","hist_median_in","pct_of_median"]]
          .to_string(float_format=lambda x: f"{x:.1f}"))

    return {
        "envelopes":    envelopes,
        "peak_swe":     peak_swe,
        "april1_table": april1_table,
    }


# Streamflow Analysis

def compute_monthly_volume(flow_df, station_id):
    """
    Compute monthly volumetric flow (m³) for Apr-Sep per water year.

    Daily mean flow (m³/s) × 86400 s/day = daily volume (m³).
    Sum per month = monthly volume (m³).

    Returns
    -------
    DataFrame indexed by water_year with columns [4,5,6,7,8,9]
    (month numbers). Only years with complete data for a given month
    are included (no partial months).
    """
    df = flow_df[["flow_cms"]].copy()
    df = _add_water_year(df)
    df["month"]        = df.index.month
    df["daily_vol_m3"] = df["flow_cms"] * 86400

    # Filter to forecast months only
    df = df[df["month"].isin(FORECAST_MONTHS)]

    # Count days per month per year to detect incomplete months
    day_counts = (df.groupby(["water_year","month"])
                    .size()
                    .unstack(fill_value=0))

    # Sum monthly volumes
    monthly_vol = (df.groupby(["water_year","month"])["daily_vol_m3"]
                     .sum()
                     .unstack())

    # Mask months with fewer than 25 days of data (incomplete)
    for m in FORECAST_MONTHS:
        if m in day_counts.columns and m in monthly_vol.columns:
            monthly_vol[m] = monthly_vol[m].where(day_counts[m] >= 25)

    # Drop the current partial water year from historical stats
    monthly_vol = monthly_vol[monthly_vol.index < 2025]

    return monthly_vol


def compute_flow_envelope(monthly_vol):
    """
    Compute historical flow envelope for each forecast month.

    Returns
    -------
    DataFrame indexed by month with columns:
        median, mean, q25, q75, min, max, n_years
    """
    records = []
    for m in FORECAST_MONTHS:
        if m not in monthly_vol.columns:
            continue
        col = monthly_vol[m].dropna()
        if len(col) < MIN_YEARS:
            print(f"  WARNING: Month {m} only has {len(col)} years of data")
            continue
        records.append({
            "month":   m,
            "median":  col.median(),
            "mean":    col.mean(),
            "q25":     col.quantile(0.25),
            "q75":     col.quantile(0.75),
            "min":     col.min(),
            "max":     col.max(),
            "n_years": len(col),
        })
    return pd.DataFrame(records).set_index("month")


def compute_april1_flow_context(flow_df):
    """
    Compute streamflow context around April 1 2025:
    - 30-day mean flow leading up to April 1
    - Historical April mean flow
    - Percent of historical median
    """
    df = flow_df[["flow_cms"]].copy()

    # 30-day window ending April 1 2025
    window = df.loc["2025-03-02":"2025-04-01", "flow_cms"].dropna()
    current_mean = window.mean() if len(window) > 0 else np.nan

    # Historical April mean flow (all years except 2025)
    hist_april = df[
        (df.index.month == 4) &
        (df.index.year < 2025)
    ]["flow_cms"].dropna()

    hist_median = hist_april.median() if len(hist_april) >= 10 else np.nan
    pct_of_median = (current_mean / hist_median * 100
                     if hist_median and hist_median > 0 else np.nan)

    return {
        "current_30day_mean_cms": current_mean,
        "hist_april_median_cms":  hist_median,
        "pct_of_median":          pct_of_median,
    }


def run_streamflow_analysis(data):
    """
    Run all streamflow analyses. Returns dict with keys:
        monthly_vol     : DataFrame (water_year × month)
        flow_envelope   : DataFrame (month × stats)
        april1_context  : dict
        monthly_vol_2025: Series (month → m³) for WY2025 through Apr 1
    """
    inlet_df = data["inlet_df"]

    print("\n--- Streamflow Analysis ---")

    monthly_vol   = compute_monthly_volume(inlet_df, "inlet")
    flow_envelope = compute_flow_envelope(monthly_vol)

    print("\nHistorical flow envelope (median monthly volume, m³):")
    print(flow_envelope["median"].apply(lambda x: f"{x:,.0f} m³"))

    april1_context = compute_april1_flow_context(inlet_df)
    print(f"\nApril 1 2025 context (inlet):")
    print(f"  30-day mean flow:        {april1_context['current_30day_mean_cms']:.1f} cms")
    print(f"  Historical April median: {april1_context['hist_april_median_cms']:.1f} cms")
    print(f"  % of median:             {april1_context['pct_of_median']:.1f}%")

    # WY2025 actual monthly volumes (partial year through April 1)
    df_2025 = inlet_df[["flow_cms"]].copy()
    df_2025 = _add_water_year(df_2025)
    df_2025["month"] = df_2025.index.month
    df_2025["daily_vol_m3"] = df_2025["flow_cms"] * 86400
    wy2025 = df_2025[
        (df_2025["water_year"] == 2025) &
        (df_2025["month"].isin(FORECAST_MONTHS))
    ]
    monthly_vol_2025 = wy2025.groupby("month")["daily_vol_m3"].sum()

    return {
        "monthly_vol":      monthly_vol,
        "flow_envelope":    flow_envelope,
        "april1_context":   april1_context,
        "monthly_vol_2025": monthly_vol_2025,
    }


# Peak SWE vs Monthly Flow

def compute_peak_swe_vs_monthly_flow(peak_swe_dict, monthly_vol):
    """
    For each station, pair peak SWE per water year with
    Apr-Sep monthly volumes for that same year.

    Returns
    -------
    dict: {station_code: DataFrame}
        Each DataFrame has columns:
            water_year, peak_swe_m, peak_swe_in, 4, 5, 6, 7, 8, 9
        Only years where both peak SWE and flow data exist are included.
    """
    paired = {}

    for code, peak_df in peak_swe_dict.items():
        # Merge peak SWE with monthly volumes on water year
        merged = peak_df.merge(monthly_vol, on="water_year", how="inner")
        merged["peak_swe_in"] = merged["peak_swe_m"] * 39.3701

        # Drop rows missing either peak SWE or any flow month
        merged = merged.dropna(subset=["peak_swe_m"])

        if len(merged) < MIN_YEARS:
            print(f"  WARNING: {code} only has {len(merged)} paired years — skipping")
            continue

        paired[code] = merged
        print(f"  {code}: {len(merged)} paired water years")

    return paired


def run_parity_analysis(swe_results, flow_results):
    """
    Build the paired peak SWE vs monthly flow dataset.
    Returns dict {station_code: paired_df}.
    """
    print("\n--- Peak SWE vs Monthly Flow Analysis ---")
    paired = compute_peak_swe_vs_monthly_flow(
        swe_results["peak_swe"],
        flow_results["monthly_vol"]
    )
    return paired


# Master runner

def analyze_all(data):
    """
    Run all analyses. Returns a single results dict containing
    everything plotting.py needs.
    """
    swe_results  = run_swe_analysis(data)
    flow_results = run_streamflow_analysis(data)
    paired       = run_parity_analysis(swe_results, flow_results)

    return {
        # SWE
        "envelopes":        swe_results["envelopes"],
        "peak_swe":         swe_results["peak_swe"],
        "april1_table":     swe_results["april1_table"],
        # Streamflow
        "monthly_vol":      flow_results["monthly_vol"],
        "flow_envelope":    flow_results["flow_envelope"],
        "april1_context":   flow_results["april1_context"],
        "monthly_vol_2025": flow_results["monthly_vol_2025"],
        # Parity
        "paired":           paired,
        # Pass through raw data for plotting
        "stations_gdf":     data["stations_gdf"],
        "inlet_df":         data["inlet_df"],
        "outlet_df":        data["outlet_df"],
        "snotel_data":      data["snotel_data"],
    }


if __name__ == "__main__":
    from data_acquisition import acquire_all
    data    = acquire_all()
    results = analyze_all(data)
    print("\n=== Analysis complete ===")