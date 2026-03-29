"""
plotting.py

Produces all figures for the Flaming Gorge water supply forecast report:

  Figure 1 - Map: basin boundary, SNOTEL stations, USGS gauges
  Figure 2 - SWE envelopes: historical range + WY2025 trace per station
  Figure 3 - Streamflow: historical monthly volume range (2x3 subplots)
  Figure 4 - Parity plots: peak SWE vs monthly flow (2x3 subplots)

All figures saved to figures/ directory as high-resolution PNGs.
Uses Agg backend so no display required (runs headless on CHPC).

Author: Magnus Tveit
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from scipy import stats
import sys
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# Config
FIGURES_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
ANALYSIS_DATE = pd.Timestamp("2025-04-01")
MONTH_NAMES   = {4:"April", 5:"May", 6:"June",
                 7:"July",  8:"August", 9:"September"}

# Color palette — consistent across all figures
C_MEDIAN   = "#2166ac"   # blue  — historical median
C_ENVELOPE = "#92c5de"   # light blue — IQR fill
C_RANGE    = "#d1e5f0"   # very light blue — min/max fill
C_WY2025   = "#d73027"   # red — current year
C_SNOTEL   = "#1a9850"   # green — SNOTEL markers
C_INLET    = "#2166ac"   # blue — inlet gauge
C_OUTLET   = "#d73027"   # red — outlet gauge


def _make_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# Figure 1: Map

def plot_map(data):
    """
    Map showing:
      - Watershed boundary
      - SNOTEL station locations (green triangles)
      - Inlet gauge 09217000 (blue circle)
      - Outlet gauge 09234500 (red circle)
    Basemap from contextily (OpenStreetMap tiles).
    Saved to figures/fig1_map.png
    """
    print("  Plotting Figure 1: Map...")

    basin_gdf    = data["basin_gdf"].to_crs(epsg=3857)
    stations_gdf = data["stations_gdf"].to_crs(epsg=3857)

    # Gauge locations from NLDI
    from dataretrieval import nwis
    inlet_info,  _ = nwis.get_info(sites="09217000")
    outlet_info, _ = nwis.get_info(sites="09234500")

    inlet_gdf = gpd.GeoDataFrame(
        inlet_info,
        geometry=gpd.points_from_xy(
            inlet_info["dec_long_va"], inlet_info["dec_lat_va"]),
        crs=4326
    ).to_crs(epsg=3857)

    outlet_gdf = gpd.GeoDataFrame(
        outlet_info,
        geometry=gpd.points_from_xy(
            outlet_info["dec_long_va"], outlet_info["dec_lat_va"]),
        crs=4326
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Basin boundary
    basin_gdf.boundary.plot(ax=ax, color="navy", linewidth=2, zorder=3)
    basin_gdf.plot(ax=ax, color="navy", alpha=0.08, zorder=2)

    # SNOTEL stations
    stations_gdf.plot(ax=ax, color=C_SNOTEL, marker="^",
                      markersize=80, zorder=5, label="SNOTEL Station")
    for code, row in stations_gdf.iterrows():
        ax.annotate(
            row["name"].split()[0],   # first word of name to keep labels short
            xy=(row.geometry.x, row.geometry.y),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color="darkgreen", zorder=6
        )

    # Gauges
    inlet_gdf.plot(ax=ax, color=C_INLET, marker="o",
                   markersize=120, zorder=5, label="Inlet Gauge (09217000)")
    outlet_gdf.plot(ax=ax, color=C_OUTLET, marker="o",
                    markersize=120, zorder=5, label="Outlet Gauge (09234500)")

    # Basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap, zoom=8, zorder=1)
    except Exception:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain, zoom=8, zorder=1)
        except Exception:
            print("    WARNING: Could not load basemap tiles — saving without basemap")

    ax.set_title("Green River Basin above Flaming Gorge Reservoir\n"
                 "SNOTEL Stations and USGS Stream Gauges",
                 fontsize=13, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    path = os.path.join(FIGURES_DIR, "fig1_map.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 2: SWE Envelopes

def _get_wy2025_trace(snotel_df):
    """
    Extract WY2025 SWE trace (Oct 2024 – Apr 1 2025)
    as a day-of-water-year indexed Series.
    """
    df = snotel_df[["WTEQ_m"]].copy()
    wy_start = pd.Timestamp("2024-10-01")
    wy2025 = df[(df.index >= wy_start) & (df.index <= ANALYSIS_DATE)].copy()
    wy2025["dowy"] = [(d - wy_start).days + 1 for d in wy2025.index]
    return wy2025.set_index("dowy")["WTEQ_m"]


def plot_swe_envelopes(results, data):
    """
    One subplot per SNOTEL station showing:
      - Min/max shaded range (light blue)
      - 25th/75th IQR shaded (medium blue)
      - Historical median line (dark blue)
      - WY2025 trace (red)
      - April 1 2025 marker (red dot)
    Layout: 2 rows × 5 cols (10 stations).
    Saved to figures/fig2_swe_envelopes.png
    """
    print("  Plotting Figure 2: SWE Envelopes...")

    envelopes    = results["envelopes"]
    snotel_data  = data["snotel_data"]
    stations_gdf = data["stations_gdf"]
    april1_table = results["april1_table"]

    codes = list(envelopes.keys())
    n     = len(codes)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             sharey=False)
    axes = axes.flatten()

    for i, code in enumerate(codes):
        ax  = axes[i]
        env = envelopes[code]
        x   = env.index   # day-of-water-year

        # Shaded ranges
        ax.fill_between(x, env["min"],  env["max"],
                        color=C_RANGE,    alpha=0.5, label="Min–Max")
        ax.fill_between(x, env["q25"],  env["q75"],
                        color=C_ENVELOPE, alpha=0.7, label="25th–75th pct")

        # Median line
        ax.plot(x, env["median"], color=C_MEDIAN,
                lw=2, label="Historical median")

        # WY2025 trace
        if code in snotel_data:
            trace = _get_wy2025_trace(snotel_data[code])
            ax.plot(trace.index, trace.values,
                    color=C_WY2025, lw=2, label="WY2025")

            # April 1 marker (dowy 183 for WY starting Oct 1)
            apr1_dowy = (ANALYSIS_DATE - pd.Timestamp("2024-10-01")).days + 1
            apr1_val  = april1_table.loc[code, "swe_2025_m"] if code in april1_table.index else np.nan
            pct       = april1_table.loc[code, "pct_of_median"] if code in april1_table.index else np.nan
            if not np.isnan(apr1_val):
                ax.plot(apr1_dowy, apr1_val, "o",
                        color=C_WY2025, markersize=8, zorder=6)
                ax.axvline(apr1_dowy, color=C_WY2025,
                           lw=1, linestyle="--", alpha=0.5)

        # Labels
        name = (stations_gdf.loc[code, "name"]
                if code in stations_gdf.index else code)
        elev = (stations_gdf.loc[code, "elevation_m"]
                if code in stations_gdf.index else "")
        pct_str = (f"\n{pct:.0f}% of median" if not np.isnan(pct) else "")
        ax.set_title(f"{name}\n{elev:.0f} m{pct_str}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Day of water year", fontsize=7)
        ax.set_ylabel("SWE (m)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(1, 275)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Single shared legend
    handles = [
        mpatches.Patch(color=C_RANGE,    alpha=0.5,  label="Min–Max range"),
        mpatches.Patch(color=C_ENVELOPE, alpha=0.7,  label="25th–75th percentile"),
        plt.Line2D([0],[0], color=C_MEDIAN, lw=2,    label="Historical median"),
        plt.Line2D([0],[0], color=C_WY2025, lw=2,    label="WY2025"),
        plt.Line2D([0],[0], color=C_WY2025, marker="o",
                   lw=0, markersize=8,                label="April 1, 2025"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Historical SWE Envelopes — Green River Basin SNOTEL Stations\n"
                 "WY2025 trace shown in red through April 1, 2025",
                 fontsize=13, fontweight="bold", y=1.01)

    path = os.path.join(FIGURES_DIR, "fig2_swe_envelopes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 3: Monthly Streamflow Range

def plot_streamflow_range(results):
    """
    2×3 subplot grid (one per forecast month Apr–Sep) showing:
      - Historical boxplot of monthly volume
      - WY2025 value as red horizontal line
      - Historical median as blue dashed line
    y-axis in million m³ for readability.
    Saved to figures/fig3_streamflow_range.png
    """
    print("  Plotting Figure 3: Streamflow Monthly Range...")

    monthly_vol      = results["monthly_vol"]
    monthly_vol_2025 = results["monthly_vol_2025"]
    flow_envelope    = results["flow_envelope"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    months = [4, 5, 6, 7, 8, 9]

    for i, m in enumerate(months):
        ax = axes[i]

        if m not in monthly_vol.columns:
            ax.set_title(MONTH_NAMES[m])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Historical distribution — convert to million m³
        hist = monthly_vol[m].dropna() / 1e6

        # Boxplot
        bp = ax.boxplot(hist, positions=[0], widths=0.5,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="gray", alpha=0.5),
                        boxprops=dict(facecolor=C_ENVELOPE, color=C_MEDIAN),
                        medianprops=dict(color=C_MEDIAN, lw=2),
                        whiskerprops=dict(color=C_MEDIAN),
                        capprops=dict(color=C_MEDIAN))

        # Scatter all historical years lightly
        ax.scatter(np.zeros(len(hist)) + np.random.uniform(-0.15, 0.15, len(hist)),
                   hist, color=C_MEDIAN, alpha=0.25, s=15, zorder=2)

        # WY2025 value
        if m in monthly_vol_2025.index:
            val_2025 = monthly_vol_2025[m] / 1e6
            ax.axhline(val_2025, color=C_WY2025, lw=2.5,
                       linestyle="-", label=f"WY2025: {val_2025:.1f} M m³")
            ax.text(0.97, val_2025, f" WY2025\n {val_2025:.1f} M m³",
                    transform=ax.get_yaxis_transform(),
                    color=C_WY2025, fontsize=8, va="center", ha="right")

        # Median line
        if m in flow_envelope.index:
            med = flow_envelope.loc[m, "median"] / 1e6
            ax.axhline(med, color=C_MEDIAN, lw=1.5,
                       linestyle="--", alpha=0.7,
                       label=f"Median: {med:.1f} M m³")

        ax.set_title(f"{MONTH_NAMES[m]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Monthly Volume (million m³)", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)

        # Add n_years annotation
        n = len(hist)
        ax.text(0.02, 0.97, f"n = {n} years",
                transform=ax.transAxes, fontsize=7,
                va="top", color="gray")

    fig.suptitle("Historical Monthly Streamflow Volume — Green River near Green River, WY (09217000)\n"
                 "April–September, WY1980–WY2024",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig3_streamflow_range.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 4: Parity Plots

def plot_parity(results, data):
    """
    One 2×3 figure per SNOTEL station (saved as fig4_{code}_parity.png).
    Each subplot: peak SWE (x) vs monthly volume Apr-Sep (y).
    Includes:
      - Scatter of historical years (color = water year)
      - Linear regression line + R² annotation
      - WY2025 peak SWE as vertical red line
      - Predicted flow range shaded
    """
    print("  Plotting Figure 4: Parity plots...")

    paired       = results["paired"]
    april1_table = results["april1_table"]
    stations_gdf = data["stations_gdf"]
    months       = [4, 5, 6, 7, 8, 9]

    for code, df in paired.items():
        name = (stations_gdf.loc[code, "name"]
                if code in stations_gdf.index else code)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        # WY2025 peak SWE for this station
        peak_2025 = (april1_table.loc[code, "swe_2025_in"]
                     if code in april1_table.index else np.nan)

        for i, m in enumerate(months):
            ax = axes[i]

            if m not in df.columns:
                ax.set_visible(False)
                continue

            x = df["peak_swe_in"].values           # peak SWE in inches
            y = df[m].values / 1e6                  # monthly vol in million m³
            valid = ~(np.isnan(x) | np.isnan(y))
            x, y = x[valid], y[valid]
            years = df["water_year"].values[valid]

            if len(x) < 5:
                ax.set_visible(False)
                continue

            # Scatter colored by year
            sc = ax.scatter(x, y, c=years, cmap="viridis",
                            s=40, alpha=0.7, zorder=3)

            # Linear regression
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=C_MEDIAN, lw=2, zorder=4)

            # Prediction interval (95%)
            n    = len(x)
            xbar = x.mean()
            s_err = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
            t_val = 1.96
            pred_se = s_err * np.sqrt(1 + 1/n + (x_line - xbar)**2 /
                                      np.sum((x - xbar)**2))
            ax.fill_between(x_line,
                            y_line - t_val * pred_se,
                            y_line + t_val * pred_se,
                            color=C_ENVELOPE, alpha=0.35, zorder=2)

            # R² annotation
            ax.text(0.05, 0.95, f"R² = {r**2:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    va="top", color=C_MEDIAN, fontweight="bold")

            # WY2025 vertical line
            if not np.isnan(peak_2025):
                ax.axvline(peak_2025, color=C_WY2025,
                           lw=2, linestyle="--", zorder=5,
                           label=f"WY2025 peak SWE")
                # Predicted value
                pred_val = slope * peak_2025 + intercept
                ax.plot(peak_2025, pred_val, "D",
                        color=C_WY2025, markersize=9, zorder=6)
                ax.text(peak_2025 + 0.3, pred_val,
                        f" {pred_val:.1f} M m³",
                        color=C_WY2025, fontsize=8, va="center")

            ax.set_title(MONTH_NAMES[m], fontsize=10, fontweight="bold")
            ax.set_xlabel("Peak SWE (inches)", fontsize=8)
            ax.set_ylabel("Monthly Volume (million m³)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)

        # Colorbar for year
        fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02,
                     label="Water Year").ax.tick_params(labelsize=8)

        fig.suptitle(f"Peak SWE vs Monthly Streamflow Volume\n"
                     f"SNOTEL: {name} ({code}) — Inlet: Green River (09217000)",
                     fontsize=12, fontweight="bold")

        path = os.path.join(FIGURES_DIR, f"fig4_{code}_parity.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {path}")


# Master runner

def plot_all(results, data):
    """Produce all figures."""
    _make_figures_dir()
    print("\n=== Plotting ===")
    plot_map(data)
    plot_swe_envelopes(results, data)
    plot_streamflow_range(results)
    plot_parity(results, data)
    print("\n=== All figures saved ===")


if __name__ == "__main__":
    from data_acquisition import acquire_all
    from analysis import analyze_all
    data    = acquire_all()
    results = analyze_all(data)
    plot_all(results, data)