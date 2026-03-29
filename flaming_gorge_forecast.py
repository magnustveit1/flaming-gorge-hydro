"""
flaming_gorge_forecast.py

Master script for the Flaming Gorge Reservoir water supply forecast.
This is the single entry point as all configuration lives here.
Calls data_acquisition, analysis, and plotting modules in sequence.


Author: Magnus Tveit
"""

import os
import sys
import pandas as pd

# Add supporting_scripts to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "supporting_scripts"))

from data_acquisition import acquire_all
from analysis import analyze_all
from plotting import plot_all

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG - all project-specific settings live here
# ══════════════════════════════════════════════════════════════════════════════

# USGS station IDs and names
INLET_ID    = "09217000"
INLET_NAME  = "Green River near Green River, WY"
OUTLET_ID   = "09234500"
OUTLET_NAME = "Green River near Greendale, UT (Flaming Gorge Dam)"

# Names for map labels and titles
BASIN_NAME     = "Green River Basin, WY/UT"
RESERVOIR_NAME = "Flaming Gorge Reservoir"

# Map display settings
MAP_ZOOM      = 8     # Zoom level for basin map
MAP_PADDING_X = 0.25  # Horizontal extent padding around basin for map (fraction of basin width)
MAP_PADDING_Y = 0.08  # Vertical extent padding around basin for map (fraction of basin height)

# Analysis date - the forecast is made as of this date
ANALYSIS_DATE = pd.Timestamp("2025-04-01")

# Historical record start
START_DATE = "1980-01-01"

# Paths - all relative to repo root
DATA_DIR    = os.path.join(ROOT_DIR, "data")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print(f" {RESERVOIR_NAME} - Water Supply Forecast")
    print(f" Analysis date: {ANALYSIS_DATE.date()}")
    print(f" Inlet gauge:   {INLET_ID}")
    print(f" Outlet gauge:  {OUTLET_ID}")
    print("=" * 60)

    # Step 1: Acquire all data
    data = acquire_all(
        inlet_id   = INLET_ID,
        outlet_id  = OUTLET_ID,
        data_dir   = DATA_DIR,
        start_date = START_DATE,
        end_date   = ANALYSIS_DATE.strftime("%Y-%m-%d"),
    )

    # Step 2: Run all analyses
    results = analyze_all(
        data          = data,
        analysis_date = ANALYSIS_DATE,
    )

    # Step 3: Produce all figures
    plot_all(
        results        = results,
        data           = data,
        inlet_id       = INLET_ID,
        outlet_id      = OUTLET_ID,
        inlet_name     = INLET_NAME,
        outlet_name    = OUTLET_NAME,
        reservoir_name = RESERVOIR_NAME,
        basin_name     = BASIN_NAME,
        map_zoom       = MAP_ZOOM,
        map_padding_x  = MAP_PADDING_X,
        map_padding_y  = MAP_PADDING_Y,
        analysis_date  = ANALYSIS_DATE,
        figures_dir    = FIGURES_DIR,
    )

    print("\n" + "=" * 60)
    print(" Forecast complete.")
    print(f" Figures saved to: {FIGURES_DIR}")
    print("=" * 60)