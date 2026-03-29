"""
plotting.py

Figure generation functions for reservoir water supply forecasting.
Produces basin maps, SWE envelope plots, streamflow distribution plots,
and peak SWE vs streamflow parity plots. All configuration passed in
as function arguments.

Figures produced:
    fig1_map.png              Basin map with SNOTEL and gauge locations
    fig2_swe_envelopes.png    Historical SWE envelopes per SNOTEL station
    fig3_streamflow_range.png Monthly streamflow volume boxplots (Apr-Sep)
    fig4_{code}_parity.png    Peak SWE vs monthly flow per SNOTEL station

Author: Magnus Tveit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from adjustText import adjust_text
from scipy import stats
from dataretrieval import nwis
from pynhd import NLDI
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

MONTH_NAMES = {4:"April", 5:"May", 6:"June",
               7:"July",  8:"August", 9:"September"}

C_MEDIAN   = "#2166ac"
C_ENVELOPE = "#92c5de"
C_RANGE    = "#d1e5f0"
C_WY2025   = "#d73027"
C_SNOTEL   = "#1a9850"
C_INLET    = "#1a3e6e"
C_OUTLET   = "#c2185b"


def _make_figures_dir(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)


# Figure 1: Map

def plot_map(data, inlet_id, outlet_id, inlet_name, outlet_name,
             reservoir_name, basin_name, map_zoom, map_padding_x,
             map_padding_y, figures_dir):
    print("  Plotting Figure 1: Map...")

    import io
    import mercantile
    import requests
    from PIL import Image
    from matplotlib.image import AxesImage

    # Reproject all layers
    inlet_basin  = data["basin_gdf"].to_crs(epsg=3857)
    outlet_basin = NLDI().get_basins(outlet_id).to_crs(epsg=3857)
    stations_gdf = data["stations_gdf"].to_crs(epsg=3857)

    inlet_info,  _ = nwis.get_info(sites=inlet_id)
    outlet_info, _ = nwis.get_info(sites=outlet_id)

    inlet_gdf = gpd.GeoDataFrame(
        inlet_info,
        geometry=gpd.points_from_xy(
            inlet_info["dec_long_va"], inlet_info["dec_lat_va"]),
        crs=4326).to_crs(epsg=3857)

    outlet_gdf = gpd.GeoDataFrame(
        outlet_info,
        geometry=gpd.points_from_xy(
            outlet_info["dec_long_va"], outlet_info["dec_lat_va"]),
        crs=4326).to_crs(epsg=3857)

    # Compute extent
    minx, miny, maxx, maxy = outlet_basin.total_bounds
    dx = (maxx - minx) * map_padding_x
    dy = (maxy - miny) * map_padding_y
    target_xlim = (minx - dx, maxx + dx)
    target_ylim = (miny - dy, maxy + dy)

    # Figure layout in inches
    FIG_WIDTH    = 14.0
    MARGIN_SIDE  = 0.5
    TOP_SPACE    = 1.1
    BOTTOM_SPACE = 1.6
    GAP          = 0.15

    map_width_in  = FIG_WIDTH - 2 * MARGIN_SIDE
    x_range       = target_xlim[1] - target_xlim[0]
    y_range       = target_ylim[1] - target_ylim[0]
    aspect        = y_range / x_range
    map_height_in = map_width_in * aspect
    FIG_HEIGHT    = TOP_SPACE + GAP + map_height_in + GAP + BOTTOM_SPACE

    # Exact fractions
    ax_left   = MARGIN_SIDE / FIG_WIDTH
    ax_bottom = (BOTTOM_SPACE + GAP) / FIG_HEIGHT
    ax_width  = map_width_in / FIG_WIDTH
    ax_height = map_height_in / FIG_HEIGHT

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
    ax_map = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # Draw basemap using contextily into ax_map
    try:
        ctx.add_basemap(ax_map, source=ctx.providers.OpenTopoMap,
                        zoom=map_zoom, zorder=0, attribution=False)
    except Exception:
        print("    WARNING: Could not load basemap tiles")

    # Force position back immediately
    ax_map.set_position([ax_left, ax_bottom, ax_width, ax_height])
    ax_map.set_xlim(target_xlim)
    ax_map.set_ylim(target_ylim)

    # Draw outlet basin
    for geom in outlet_basin.geometry:
        x, y = geom.exterior.xy
        ax_map.fill(x, y, color=C_OUTLET, alpha=0.10, zorder=1)
        ax_map.plot(x, y, color=C_OUTLET, lw=2.5, linestyle="--", zorder=2)

    # Draw inlet basin
    for geom in inlet_basin.geometry:
        x, y = geom.exterior.xy
        ax_map.fill(x, y, color=C_INLET, alpha=0.15, zorder=3)
        ax_map.plot(x, y, color=C_INLET, lw=2.5, linestyle="-", zorder=4)

    # SNOTEL stations
    stations_gdf.plot(ax=ax_map, color=C_SNOTEL, marker="^",
                      markersize=120, zorder=6,
                      edgecolor="white", linewidth=1.0)

    texts, pt_x, pt_y = [], [], []
    for code, row in stations_gdf.iterrows():
        name_short = row["name"].split()[0]
        pt_x.append(row.geometry.x)
        pt_y.append(row.geometry.y)
        t = ax_map.text(
            row.geometry.x, row.geometry.y, name_short,
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=C_SNOTEL,
                      alpha=0.85, edgecolor="none"),
            zorder=8)
        texts.append(t)

    adjust_text(
        texts,
        x=np.array(pt_x),
        y=np.array(pt_y),
        ax=ax_map,
        expand_points=(1.8, 1.8),
        expand_text=(1.3, 1.3),
        arrowprops=dict(arrowstyle="-", color=C_SNOTEL,
                        lw=1.0, alpha=0.7,
                        shrinkA=5, shrinkB=5))

    # Gauges
    inlet_gdf.plot(ax=ax_map, color=C_INLET, marker="o",
                   markersize=180, zorder=7,
                   edgecolor="white", linewidth=2.0)
    outlet_gdf.plot(ax=ax_map, color=C_OUTLET, marker="o",
                    markersize=180, zorder=7,
                    edgecolor="white", linewidth=2.0)

    for _, row in inlet_gdf.iterrows():
        ax_map.annotate(
            f"Inlet ({inlet_id})",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(16, -20), textcoords="offset points",
            fontsize=11, fontweight="bold", color=C_INLET,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"),
            zorder=9)

    for _, row in outlet_gdf.iterrows():
        ax_map.annotate(
            f"Outlet ({outlet_id})",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(16, -20), textcoords="offset points",
            fontsize=11, fontweight="bold", color=C_OUTLET,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"),
            zorder=9)

    # Set extent
    ax_map.set_xlim(target_xlim)
    ax_map.set_ylim(target_ylim)

    # Basemap (AFTER vector layers)
    try:
        ctx.add_basemap(ax_map, source=ctx.providers.OpenTopoMap,
                        zoom=map_zoom, zorder=0, attribution=False)
    except Exception:
        print("    WARNING: Could not load basemap tiles")

    # Force position and extent back after messes it up contextily
    ax_map.set_position([ax_left, ax_bottom, ax_width, ax_height])
    ax_map.set_xlim(target_xlim)
    ax_map.set_ylim(target_ylim)
    ax_map.set_axis_off()

    # Title
    title_y = (BOTTOM_SPACE + GAP + map_height_in + GAP +
                TOP_SPACE * 0.5) / FIG_HEIGHT
    fig.text(
        0.5, title_y,
        f"{reservoir_name} - {basin_name}\n"
        "SNOTEL Stations and USGS Stream Gauges",
        ha="center", va="center",
        fontsize=17, fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(facecolor=C_INLET,  edgecolor=C_INLET,
                       alpha=0.5,
                       label=f"Inlet drainage basin ({inlet_id})"),
        mpatches.Patch(facecolor=C_OUTLET, edgecolor=C_OUTLET,
                       alpha=0.4,
                       label=f"Outlet drainage basin ({outlet_id})"),
        plt.Line2D([0],[0], color=C_INLET,  marker="o",
                   lw=0, markersize=12,
                   label=f"Inlet gauge - {inlet_name}"),
        plt.Line2D([0],[0], color=C_OUTLET, marker="o",
                   lw=0, markersize=12,
                   label=f"Outlet gauge - {outlet_name}"),
        plt.Line2D([0],[0], color=C_SNOTEL, marker="^",
                   lw=0, markersize=12,
                   label="Inlet drainage SNOTEL sites"),
    ]
    legend_y = (BOTTOM_SPACE * 0.45) / FIG_HEIGHT
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        fontsize=12,
        framealpha=0.95,
        edgecolor="lightgray")

    # Attribution
    fig.text(
        0.5, 0.01,
        "Basemap: © OpenTopoMap contributors, © OpenStreetMap contributors  |  "
        "Watershed: USGS NHD via NLDI  |  "
        "SNOTEL: NRCS via egagli/snotel_ccss_stations",
        ha="center", va="bottom",
        fontsize=9, color="gray", style="italic")

    path = os.path.join(figures_dir, "fig1_map.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 2: SWE Envelopes

def _get_current_wy_trace(snotel_df, analysis_date):
    """
    Extract current water year SWE trace from Oct 1 through analysis_date.
    Returns a Series indexed by day-of-water-year.
    """
    wy_start = pd.Timestamp(f"{analysis_date.year - 1}-10-01")
    df = snotel_df[["WTEQ_m"]].copy()
    wy = df[(df.index >= wy_start) & (df.index <= analysis_date)].copy()
    wy["dowy"] = [(d - wy_start).days + 1 for d in wy.index]
    return wy.set_index("dowy")["WTEQ_m"]


def plot_swe_envelopes(results, data, analysis_date, basin_name, figures_dir):
    """
    One subplot per SNOTEL station showing historical SWE envelope
    and current water year trace.

    Layout: 2 rows x 5 columns (10 stations).
    Shaded bands show min/max and 25th/75th percentile ranges.
    Current year shown in red with analysis date marker.

    Parameters
    ----------
    results : dict        Output of analyze_all()
    data : dict           Output of acquire_all()
    analysis_date : pd.Timestamp
    basin_name : str      Basin name for figure title
    figures_dir : str
    """
    print("  Plotting Figure 2: SWE Envelopes...")

    envelopes    = results["envelopes"]
    snotel_data  = data["snotel_data"]
    stations_gdf = data["stations_gdf"]
    april1_table = results["april1_table"]

    codes = list(envelopes.keys())
    ncols = 5
    nrows = int(np.ceil(len(codes) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             sharey=False)
    axes = axes.flatten()

    for i, code in enumerate(codes):
        ax  = axes[i]
        env = envelopes[code]
        x   = env.index

        ax.fill_between(x, env["min"], env["max"],
                        color=C_RANGE, alpha=0.5)
        ax.fill_between(x, env["q25"], env["q75"],
                        color=C_ENVELOPE, alpha=0.7)
        ax.plot(x, env["median"], color=C_MEDIAN, lw=2)

        if code in snotel_data:
            trace = _get_current_wy_trace(snotel_data[code], analysis_date)
            ax.plot(trace.index, trace.values, color=C_WY2025, lw=2)

            wy_start  = pd.Timestamp(f"{analysis_date.year - 1}-10-01")
            apr1_dowy = (analysis_date - wy_start).days + 1
            apr1_val  = (april1_table.loc[code, "swe_m"]
                         if code in april1_table.index else np.nan)
            pct       = (april1_table.loc[code, "pct_of_median"]
                         if code in april1_table.index else np.nan)

            if not np.isnan(apr1_val):
                ax.plot(apr1_dowy, apr1_val, "o",
                        color=C_WY2025, markersize=8, zorder=6)
                ax.axvline(apr1_dowy, color=C_WY2025,
                           lw=1, linestyle="--", alpha=0.5)

        name    = (stations_gdf.loc[code, "name"]
                   if code in stations_gdf.index else code)
        elev    = (stations_gdf.loc[code, "elevation_m"]
                   if code in stations_gdf.index else "")
        pct_str = (f"\n{pct:.0f}% of median"
                   if code in april1_table.index
                   and not np.isnan(april1_table.loc[code, "pct_of_median"])
                   else "")

        ax.set_title(f"{name}\n{elev:.0f} m{pct_str}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Day of water year", fontsize=7)
        ax.set_ylabel("SWE (m)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(1, 275)
        ax.set_ylim(bottom=0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color=C_RANGE,    alpha=0.5, label="Min–Max range"),
        mpatches.Patch(color=C_ENVELOPE, alpha=0.7, label="25th–75th percentile"),
        plt.Line2D([0],[0], color=C_MEDIAN, lw=2,   label="Historical median"),
        plt.Line2D([0],[0], color=C_WY2025, lw=2,
                   label=f"WY{analysis_date.year}"),
        plt.Line2D([0],[0], color=C_WY2025, marker="o",
                   lw=0, markersize=8,
                   label=f"{analysis_date.date()}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Historical SWE Envelopes - {basin_name} SNOTEL Stations\n"
        f"WY{analysis_date.year} trace shown in red through "
        f"{analysis_date.date()}",
        fontsize=13, fontweight="bold", y=1.01)

    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    path = os.path.join(figures_dir, "fig2_swe_envelopes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 3: Monthly Streamflow Range

def plot_streamflow_range(results, inlet_id, analysis_date, figures_dir):
    """
    2×3 subplot grid showing historical monthly streamflow volume
    distribution (Apr–Sep) as boxplots with current water year overlay.

    Each subplot shows:
      - Boxplot of all historical years
      - Individual year scatter (jittered)
      - Current WY value as red horizontal line
      - Historical median as dashed blue line

    Parameters
    ----------
    results : dict        Output of analyze_all()
    inlet_id : str        USGS inlet station ID (for title)
    analysis_date : pd.Timestamp
    figures_dir : str
    """
    print("  Plotting Figure 3: Streamflow Monthly Range...")

    monthly_vol     = results["monthly_vol"]
    monthly_vol_cur = results["monthly_vol_current_wy"]
    flow_envelope   = results["flow_envelope"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, m in enumerate([4, 5, 6, 7, 8, 9]):
        ax = axes[i]

        if m not in monthly_vol.columns:
            ax.set_title(MONTH_NAMES[m])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        hist = monthly_vol[m].dropna() / 1e6

        ax.boxplot(
            hist, positions=[0], widths=0.5,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor="gray", alpha=0.5),
            boxprops=dict(facecolor=C_ENVELOPE, color=C_MEDIAN),
            medianprops=dict(color=C_MEDIAN, lw=2),
            whiskerprops=dict(color=C_MEDIAN),
            capprops=dict(color=C_MEDIAN))

        ax.scatter(
            np.zeros(len(hist)) + np.random.uniform(-0.15, 0.15, len(hist)),
            hist, color=C_MEDIAN, alpha=0.25, s=15, zorder=2)

        if m in monthly_vol_cur.index:
            val = monthly_vol_cur[m] / 1e6
            ax.axhline(val, color=C_WY2025, lw=2.5)
            ax.text(0.97, val,
                    f" WY{analysis_date.year}\n {val:.1f} M m³",
                    transform=ax.get_yaxis_transform(),
                    color=C_WY2025, fontsize=8,
                    va="center", ha="right")

        if m in flow_envelope.index:
            med = flow_envelope.loc[m, "median"] / 1e6
            ax.axhline(med, color=C_MEDIAN, lw=1.5,
                       linestyle="--", alpha=0.7)

        ax.set_title(MONTH_NAMES[m], fontsize=11, fontweight="bold")
        ax.set_ylabel("Monthly Volume (million m³)", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)

        ax.text(0.02, 0.97, f"n = {len(hist)} years",
                transform=ax.transAxes, fontsize=7,
                va="top", color="gray")

    fig.suptitle(
        f"Historical Monthly Streamflow Volume - USGS {inlet_id}\n"
        f"April–September, WY1980–WY{analysis_date.year - 1}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(figures_dir, "fig3_streamflow_range.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# Figure 4: Parity Plots

def plot_parity(results, data, analysis_date, figures_dir):
    """
    One 2×3 parity figure per SNOTEL station.
    Each subplot: peak SWE (x-axis, inches) vs monthly streamflow
    volume (y-axis, million m³) for one forecast month (Apr–Sep).

    Includes:
      - Scatter of historical year pairs (colored by water year)
      - Linear regression line
      - 95% prediction interval shading
      - R² annotation
      - Vertical line at current year peak SWE with predicted value

    Parameters
    ----------
    results : dict        Output of analyze_all()
    data : dict           Output of acquire_all()
    analysis_date : pd.Timestamp
    figures_dir : str
    """
    print("  Plotting Figure 4: Parity plots...")

    paired       = results["paired"]
    april1_table = results["april1_table"]
    stations_gdf = data["stations_gdf"]

    for code, df in paired.items():
        name = (stations_gdf.loc[code, "name"]
                if code in stations_gdf.index else code)
        peak_cur = (april1_table.loc[code, "swe_in"]
                    if code in april1_table.index else np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        sc = None

        for i, m in enumerate([4, 5, 6, 7, 8, 9]):
            ax = axes[i]

            if m not in df.columns:
                ax.set_visible(False)
                continue

            x = df["peak_swe_in"].values
            y = df[m].values / 1e6
            valid = ~(np.isnan(x) | np.isnan(y))
            x, y  = x[valid], y[valid]
            years = df["water_year"].values[valid]

            if len(x) < 5:
                ax.set_visible(False)
                continue

            sc = ax.scatter(x, y, c=years, cmap="viridis",
                            s=40, alpha=0.7, zorder=3)

            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=C_MEDIAN, lw=2, zorder=4)

            n, xbar = len(x), x.mean()
            s_err   = np.sqrt(np.sum((y-(slope*x+intercept))**2) / (n-2))
            pred_se = s_err * np.sqrt(
                1 + 1/n + (x_line - xbar)**2 / np.sum((x - xbar)**2))
            ax.fill_between(x_line,
                            y_line - 1.96 * pred_se,
                            y_line + 1.96 * pred_se,
                            color=C_ENVELOPE, alpha=0.35, zorder=2)

            ax.text(0.05, 0.95, f"R² = {r**2:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    va="top", color=C_MEDIAN, fontweight="bold")

            if not np.isnan(peak_cur):
                ax.axvline(peak_cur, color=C_WY2025,
                           lw=2, linestyle="--", zorder=5)
                pred_val = slope * peak_cur + intercept
                ax.plot(peak_cur, pred_val, "D",
                        color=C_WY2025, markersize=9, zorder=6)
                ax.text(peak_cur + 0.3, pred_val,
                        f" {pred_val:.1f} M m³",
                        color=C_WY2025, fontsize=8, va="center")

            ax.set_title(MONTH_NAMES[m], fontsize=10, fontweight="bold")
            ax.set_xlabel("Peak SWE (inches)", fontsize=8)
            ax.set_ylabel("Monthly Volume (million m³)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)

        if sc is not None:
            fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02,
                         label="Water Year").ax.tick_params(labelsize=8)

        fig.suptitle(
            f"Peak SWE vs Monthly Streamflow Volume\n"
            f"SNOTEL: {name} ({code})  |  "
            f"Analysis date: {analysis_date.date()}",
            fontsize=12, fontweight="bold")

        path = os.path.join(figures_dir, f"fig4_{code}_parity.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {path}")


# Master runner

def plot_all(results, data, inlet_id, outlet_id, inlet_name, outlet_name,
             reservoir_name, basin_name, map_zoom, map_padding_x,
             map_padding_y, analysis_date, figures_dir):
    """
    Produce all figures for the forecast report.

    Parameters
    ----------
    results : dict           Output of analyze_all()
    data : dict              Output of acquire_all()
    inlet_id : str           USGS inlet station ID
    outlet_id : str          USGS outlet station ID
    inlet_name : str         Human-readable inlet gauge name
    outlet_name : str        Human-readable outlet gauge name
    reservoir_name : str     Reservoir name for titles
    basin_name : str         Basin name for titles
    map_zoom : int           Contextily tile zoom level
    map_padding_x : float    Horizontal fractional padding around basin
    map_padding_y : float    Vertical fractional padding around basin
    analysis_date : pd.Timestamp
    figures_dir : str        Output directory for all figures
    """
    _make_figures_dir(figures_dir)
    print("\n=== Plotting ===")
    plot_map(data, inlet_id, outlet_id, inlet_name, outlet_name,
             reservoir_name, basin_name, map_zoom, map_padding_x,
             map_padding_y, figures_dir)
    plot_swe_envelopes(results, data, analysis_date, basin_name, figures_dir)
    plot_streamflow_range(results, inlet_id, analysis_date, figures_dir)
    plot_parity(results, data, analysis_date, figures_dir)
    print("\n=== All figures saved ===")