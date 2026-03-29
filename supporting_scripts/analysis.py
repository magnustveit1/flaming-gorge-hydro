"""
analysis.py

Statistical analysis functions for reservoir water supply forecasting.
Computes historical SWE envelopes, peak SWE per water year, monthly
streamflow volumes, and paired peak SWE vs flow relationships for
parity analysis. All configuration passed in as function arguments.

Analyses produced:
    - Historical SWE envelope (min, max, median, IQR) by day-of-water-year
    - April 1 SWE vs historical median per SNOTEL station
    - Monthly volumetric streamflow (m³) for Apr-Sep per water year
    - Peak SWE paired with monthly flow for regression analysis

Author: Magnus Tveit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

FORECAST_MONTHS = [4, 5, 6, 7, 8, 9]
MONTH_NAMES     = {4:"April", 5:"May", 6:"June",
                   7:"July",  8:"August", 9:"September"}
MIN_YEARS       = 10


def _water_year(dt):
    return dt.year + 1 if dt.month >= 10 else dt.year


def _add_water_year(df):
    df = df.copy()
    df["water_year"] = df.index.map(_water_year)
    return df


def _doy_label(dt):
    wy_start = pd.Timestamp(f"{_water_year(dt) - 1}-10-01")
    return (dt - wy_start).days + 1


# SWE

def compute_swe_envelope(snotel_df, station_code, analysis_year):
    """
    Compute historical SWE envelope for a single station.
    Excludes the analysis water year from the historical baseline.
    """
    df = snotel_df[["WTEQ_m"]].copy()
    df = _add_water_year(df)
    historical = df[df["water_year"] < analysis_year].copy()
    historical["dowy"] = historical.index.map(_doy_label)
    historical = historical[historical["dowy"] <= 275]

    n_years = historical["water_year"].nunique()
    if n_years < MIN_YEARS:
        print(f"  WARNING: {station_code} only has {n_years} years - skipping")
        return None

    grouped = historical.groupby("dowy")["WTEQ_m"]
    return pd.DataFrame({
        "median":  grouped.median(),
        "mean":    grouped.mean(),
        "q25":     grouped.quantile(0.25),
        "q75":     grouped.quantile(0.75),
        "min":     grouped.min(),
        "max":     grouped.max(),
        "n_years": grouped.count(),
    })


def compute_peak_swe_per_year(snotel_df, station_code):
    """Peak SWE per water year for a station."""
    df = snotel_df[["WTEQ_m"]].copy()
    df = _add_water_year(df)
    df["month"] = df.index.month
    df = df[df["month"].isin([10,11,12,1,2,3,4,5,6])]
    return (df.groupby("water_year")["WTEQ_m"]
              .max()
              .dropna()
              .reset_index()
              .rename(columns={"WTEQ_m": "peak_swe_m"}))


def compute_april1_swe(snotel_data, stations_gdf, analysis_date):
    """
    April 1 SWE vs historical median for each station.

    Parameters
    ----------
    analysis_date : pd.Timestamp
    """
    records = []
    for code, df in snotel_data.items():
        df = df[["WTEQ_m"]].copy()

        # Current year value
        try:
            val = df.loc[analysis_date, "WTEQ_m"]
        except KeyError:
            nearby = df.loc[
                analysis_date - pd.Timedelta(days=3):
                analysis_date + pd.Timedelta(days=3), "WTEQ_m"
            ].dropna()
            val = nearby.iloc[-1] if len(nearby) > 0 else np.nan

        # Historical April 1 values
        hist = df[
            (df.index.month == 4) &
            (df.index.day == 1) &
            (df.index.year < analysis_date.year)
        ]["WTEQ_m"].dropna()

        med = hist.median() if len(hist) >= 5 else np.nan
        pct = (val / med * 100) if (med and med > 0) else np.nan

        name = stations_gdf.loc[code, "name"] if code in stations_gdf.index else code
        elev = stations_gdf.loc[code, "elevation_m"] if code in stations_gdf.index else np.nan

        records.append({
            "code":           code,
            "name":           name,
            "elevation_m":    elev,
            "swe_m":          val,
            "hist_median_m":  med,
            "pct_of_median":  pct,
            "swe_in":         val * 39.3701 if not np.isnan(val) else np.nan,
            "hist_median_in": med * 39.3701 if not np.isnan(med) else np.nan,
        })

    return pd.DataFrame(records).set_index("code")


def run_swe_analysis(data, analysis_date):
    """
    Run all SWE analyses.

    Parameters
    ----------
    data : dict
        Output of acquire_all().
    analysis_date : pd.Timestamp

    Returns
    -------
    dict with keys: envelopes, peak_swe, april1_table
    """
    snotel_data  = data["snotel_data"]
    stations_gdf = data["stations_gdf"]
    analysis_year = analysis_date.year

    print("\n--- SWE Analysis ---")
    envelopes, peak_swe = {}, {}

    for code, df in snotel_data.items():
        print(f"  Processing {code}...")
        env = compute_swe_envelope(df, code, analysis_year)
        if env is not None:
            envelopes[code] = env
        pk = compute_peak_swe_per_year(df, code)
        if len(pk) >= MIN_YEARS:
            peak_swe[code] = pk

    april1_table = compute_april1_swe(snotel_data, stations_gdf, analysis_date)
    print("\nApril 1 SWE summary:")
    print(april1_table[["name","swe_in","hist_median_in","pct_of_median"]]
          .to_string(float_format=lambda x: f"{x:.1f}"))

    return {"envelopes": envelopes, "peak_swe": peak_swe,
            "april1_table": april1_table}


# Streamflow

def compute_monthly_volume(flow_df, analysis_year):
    """
    Monthly volumetric flow (m³) for Apr-Sep per water year.

    Parameters
    ----------
    flow_df : DataFrame
        Streamflow with DatetimeIndex and column flow_cms.
    analysis_year : int
        Current water year - excluded from historical stats.
    """
    df = flow_df[["flow_cms"]].copy()
    df = _add_water_year(df)
    df["month"]        = df.index.month
    df["daily_vol_m3"] = df["flow_cms"] * 86400
    df = df[df["month"].isin(FORECAST_MONTHS)]

    day_counts  = (df.groupby(["water_year","month"]).size().unstack(fill_value=0))
    monthly_vol = (df.groupby(["water_year","month"])["daily_vol_m3"]
                     .sum().unstack())

    # Mask incomplete months
    for m in FORECAST_MONTHS:
        if m in day_counts.columns and m in monthly_vol.columns:
            monthly_vol[m] = monthly_vol[m].where(day_counts[m] >= 25)

    # Exclude current water year from historical
    return monthly_vol[monthly_vol.index < analysis_year]


def compute_flow_envelope(monthly_vol):
    """Historical flow envelope per forecast month."""
    records = []
    for m in FORECAST_MONTHS:
        if m not in monthly_vol.columns:
            continue
        col = monthly_vol[m].dropna()
        if len(col) < MIN_YEARS:
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


def compute_april1_flow_context(flow_df, analysis_date):
    """30-day mean flow around analysis date vs historical April median."""
    df = flow_df[["flow_cms"]].copy()
    window = df.loc[
        analysis_date - pd.Timedelta(days=30):analysis_date,
        "flow_cms"
    ].dropna()
    current_mean = window.mean() if len(window) > 0 else np.nan

    hist = df[
        (df.index.month == 4) &
        (df.index.year < analysis_date.year)
    ]["flow_cms"].dropna()

    hist_med = hist.median() if len(hist) >= 10 else np.nan
    pct = (current_mean / hist_med * 100) if hist_med else np.nan

    return {
        "current_30day_mean_cms": current_mean,
        "hist_april_median_cms":  hist_med,
        "pct_of_median":          pct,
    }


def run_streamflow_analysis(data, analysis_date):
    """
    Run all streamflow analyses.

    Parameters
    ----------
    data : dict
        Output of acquire_all().
    analysis_date : pd.Timestamp

    Returns
    -------
    dict with keys: monthly_vol, flow_envelope,
                    april1_context, monthly_vol_current_wy
    """
    inlet_df      = data["inlet_df"]
    analysis_year = analysis_date.year

    print("\n--- Streamflow Analysis ---")
    monthly_vol   = compute_monthly_volume(inlet_df, analysis_year)
    flow_envelope = compute_flow_envelope(monthly_vol)

    print("\nHistorical median monthly volume (m³):")
    print(flow_envelope["median"].apply(lambda x: f"{x:,.0f} m³"))

    april1_context = compute_april1_flow_context(inlet_df, analysis_date)
    print(f"\nContext at {analysis_date.date()}:")
    print(f"  30-day mean:     {april1_context['current_30day_mean_cms']:.1f} cms")
    print(f"  April median:    {april1_context['hist_april_median_cms']:.1f} cms")
    print(f"  % of median:     {april1_context['pct_of_median']:.1f}%")

    # Current water year volumes (partial, for plotting)
    df = inlet_df[["flow_cms"]].copy()
    df = _add_water_year(df)
    df["month"]        = df.index.month
    df["daily_vol_m3"] = df["flow_cms"] * 86400
    wy_current = df[
        (df["water_year"] == analysis_year) &
        (df["month"].isin(FORECAST_MONTHS))
    ]
    monthly_vol_current = wy_current.groupby("month")["daily_vol_m3"].sum()

    return {
        "monthly_vol":          monthly_vol,
        "flow_envelope":        flow_envelope,
        "april1_context":       april1_context,
        "monthly_vol_current_wy": monthly_vol_current,
    }


# Parity

def compute_peak_swe_vs_monthly_flow(peak_swe_dict, monthly_vol):
    """Pair peak SWE per water year with Apr-Sep monthly volumes."""
    paired = {}
    for code, peak_df in peak_swe_dict.items():
        merged = peak_df.merge(monthly_vol, on="water_year", how="inner")
        merged["peak_swe_in"] = merged["peak_swe_m"] * 39.3701
        merged = merged.dropna(subset=["peak_swe_m"])
        if len(merged) < MIN_YEARS:
            print(f"  WARNING: {code} only {len(merged)} paired years - skipping")
            continue
        paired[code] = merged
        print(f"  {code}: {len(merged)} paired water years")
    return paired


def run_parity_analysis(swe_results, flow_results):
    print("\n--- Peak SWE vs Monthly Flow ---")
    return compute_peak_swe_vs_monthly_flow(
        swe_results["peak_swe"],
        flow_results["monthly_vol"]
    )


# Master

def analyze_all(data, analysis_date):
    """
    Run all analyses.

    Parameters
    ----------
    data : dict
        Output of acquire_all().
    analysis_date : pd.Timestamp
        The forecast date (e.g. April 1 2025).

    Returns
    -------
    dict containing all results needed by plotting.py
    """
    swe_results  = run_swe_analysis(data, analysis_date)
    flow_results = run_streamflow_analysis(data, analysis_date)
    paired       = run_parity_analysis(swe_results, flow_results)

    return {
        "envelopes":              swe_results["envelopes"],
        "peak_swe":               swe_results["peak_swe"],
        "april1_table":           swe_results["april1_table"],
        "monthly_vol":            flow_results["monthly_vol"],
        "flow_envelope":          flow_results["flow_envelope"],
        "april1_context":         flow_results["april1_context"],
        "monthly_vol_current_wy": flow_results["monthly_vol_current_wy"],
        "paired":                 paired,
        "stations_gdf":           data["stations_gdf"],
        "inlet_df":               data["inlet_df"],
        "outlet_df":              data["outlet_df"],
        "snotel_data":            data["snotel_data"],
        "analysis_date":          analysis_date,
    }