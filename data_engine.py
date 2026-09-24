"""
data_engine.py
==============

Modular ETL + spatial feature engineering pipeline for a dynamic NYC
municipal waste routing dashboard.

Pipeline stages
----------------
1. Fetch      -> pull DSNY litter basket locations and 311 "Sanitation
                 Condition" complaints from NYC Open Data (Socrata).
2. Clean      -> coerce lat/lon to numeric, drop rows with missing or
                 out-of-bounds coordinates.
3. Enrich     -> use a BallTree (haversine metric) to compute, for every
                 litter basket:
                     - distance to nearest 311 complaint (m)
                     - count of complaints within 300m
                     - distance to nearest (synthetic) school (m)
                     - is_school_zone flag (within 500m of a school)
4. Save       -> write enriched litter baskets and cleaned complaints to
                 data/*.csv

Run directly:
    python data_engine.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

LITTER_BASKET_URL = "https://data.cityofnewyork.us/resource/8znf-7b2c.json"
COMPLAINTS_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

LITTER_BASKET_LIMIT = 500
COMPLAINTS_LIMIT = 1000

# Rough NYC bounding box used to sanity-filter coordinates.
NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.65

EARTH_RADIUS_M = 6_371_000.0

N_SYNTHETIC_SCHOOLS = 15
COMPLAINT_RADIUS_M = 300.0
SCHOOL_ZONE_RADIUS_M = 500.0

DATA_DIR = "data"
LITTER_BASKET_OUTFILE = os.path.join(DATA_DIR, "litter_baskets_enriched.csv")
COMPLAINTS_OUTFILE = os.path.join(DATA_DIR, "complaints_311.csv")

REQUEST_TIMEOUT_S = 30
RANDOM_SEED = 42


# --------------------------------------------------------------------------
# 1. Data Fetching
# --------------------------------------------------------------------------

def _get_json(url: str, params: dict) -> list:
    """GET a Socrata endpoint and return the parsed JSON list of records."""
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def fetch_litter_baskets(limit: int = LITTER_BASKET_LIMIT) -> pd.DataFrame:
    """Fetch DSNY litter basket locations from NYC Open Data."""
    print(f"[fetch] Requesting {limit} litter basket records ...")
    records = _get_json(LITTER_BASKET_URL, params={"$limit": limit})
    df = pd.DataFrame.from_records(records)
    print(f"[fetch] Retrieved {len(df)} litter basket rows.")
    return df


def fetch_311_complaints(limit: int = COMPLAINTS_LIMIT) -> pd.DataFrame:
    """Fetch 311 'Sanitation Condition' complaints from NYC Open Data."""
    print(f"[fetch] Requesting {limit} 311 sanitation complaint records ...")
    params = {
        "$where": "complaint_type='Sanitation Condition'",
        "$limit": limit,
    }
    records = _get_json(COMPLAINTS_URL, params=params)
    df = pd.DataFrame.from_records(records)
    print(f"[fetch] Retrieved {len(df)} 311 complaint rows.")
    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------

def clean_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Coerce lat/lon columns to numeric and drop rows with missing or
    out-of-NYC-bounds coordinates.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"Expected columns '{lat_col}'/'{lon_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=[lat_col, lon_col])

    in_bounds = (
        out[lat_col].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & out[lon_col].between(NYC_LON_MIN, NYC_LON_MAX)
    )
    out = out.loc[in_bounds].reset_index(drop=True)

    dropped = before - len(out)
    print(
        f"[clean] Dropped {dropped} rows with missing/out-of-bounds "
        f"coordinates ({len(out)} rows remain)."
    )
    return out


# --------------------------------------------------------------------------
# 3. Spatial Feature Engineering
# --------------------------------------------------------------------------

def _to_radians(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    """Build an (N, 2) radians array [lat, lon] for BallTree/haversine use."""
    return np.radians(df[[lat_col, lon_col]].to_numpy(dtype=float))


def build_ball_tree(df: pd.DataFrame, lat_col: str, lon_col: str) -> BallTree:
    """Construct a haversine BallTree over the given lat/lon points."""
    coords_rad = _to_radians(df, lat_col, lon_col)
    return BallTree(coords_rad, metric="haversine")


def nearest_distance_m(
    source_df: pd.DataFrame,
    target_tree: BallTree,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> np.ndarray:
    """
    For each row in source_df, return the great-circle distance (meters)
    to the nearest point indexed in target_tree.
    """
    source_rad = _to_radians(source_df, lat_col, lon_col)
    dist_rad, _ = target_tree.query(source_rad, k=1)
    return dist_rad[:, 0] * EARTH_RADIUS_M


def count_within_radius(
    source_df: pd.DataFrame,
    target_tree: BallTree,
    radius_m: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> np.ndarray:
    """
    For each row in source_df, count how many target-tree points fall
    within radius_m meters.
    """
    source_rad = _to_radians(source_df, lat_col, lon_col)
    radius_rad = radius_m / EARTH_RADIUS_M
    neighbor_idx_lists = target_tree.query_radius(source_rad, r=radius_rad)
    return np.array([len(idxs) for idxs in neighbor_idx_lists])


def generate_synthetic_schools(
    reference_df: pd.DataFrame,
    n_schools: int = N_SYNTHETIC_SCHOOLS,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate n_schools random lat/lon points uniformly within the bounding
    box of reference_df, standing in for a real NYC schools dataset.
    """
    rng = np.random.default_rng(seed)
    lat_min, lat_max = reference_df[lat_col].min(), reference_df[lat_col].max()
    lon_min, lon_max = reference_df[lon_col].min(), reference_df[lon_col].max()

    schools = pd.DataFrame(
        {
            "school_id": [f"SYN-SCHOOL-{i+1:02d}" for i in range(n_schools)],
            lat_col: rng.uniform(lat_min, lat_max, size=n_schools),
            lon_col: rng.uniform(lon_min, lon_max, size=n_schools),
        }
    )
    print(
        f"[enrich] Generated {n_schools} synthetic school points within "
        f"bounding box lat[{lat_min:.4f}, {lat_max:.4f}] "
        f"lon[{lon_min:.4f}, {lon_max:.4f}]."
    )
    return schools


def enrich_litter_baskets(
    baskets_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Attach spatial features to each litter basket:
      - dist_to_nearest_complaint_m
      - complaints_within_300m
      - dist_to_nearest_school_m
      - is_school_zone
    """
    enriched = baskets_df.copy()

    # --- Nearest / density features vs 311 complaints ---
    print("[enrich] Building BallTree over 311 complaints ...")
    complaint_tree = build_ball_tree(complaints_df, lat_col, lon_col)

    print("[enrich] Computing distance to nearest complaint ...")
    enriched["dist_to_nearest_complaint_m"] = nearest_distance_m(
        enriched, complaint_tree, lat_col, lon_col
    )

    print(f"[enrich] Counting complaints within {COMPLAINT_RADIUS_M:.0f}m ...")
    enriched["complaints_within_300m"] = count_within_radius(
        enriched, complaint_tree, COMPLAINT_RADIUS_M, lat_col, lon_col
    )

    # --- Synthetic schools ---
    schools_df = generate_synthetic_schools(enriched, lat_col=lat_col, lon_col=lon_col)
    school_tree = build_ball_tree(schools_df, lat_col, lon_col)

    print("[enrich] Computing distance to nearest school ...")
    enriched["dist_to_nearest_school_m"] = nearest_distance_m(
        enriched, school_tree, lat_col, lon_col
    )
    enriched["is_school_zone"] = (
        enriched["dist_to_nearest_school_m"] <= SCHOOL_ZONE_RADIUS_M
    )

    return enriched


# --------------------------------------------------------------------------
# 4. Output
# --------------------------------------------------------------------------

def save_outputs(baskets_df: pd.DataFrame, complaints_df: pd.DataFrame) -> None:
    """Write enriched litter baskets and cleaned complaints to data/*.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    baskets_df.to_csv(LITTER_BASKET_OUTFILE, index=False)
    complaints_df.to_csv(COMPLAINTS_OUTFILE, index=False)
    print(f"[save] Wrote {len(baskets_df)} rows -> {LITTER_BASKET_OUTFILE}")
    print(f"[save] Wrote {len(complaints_df)} rows -> {COMPLAINTS_OUTFILE}")


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------

def run_pipeline() -> Optional[pd.DataFrame]:
    """Execute the full fetch -> clean -> enrich -> save pipeline."""
    print("=" * 70)
    print("NYC Waste Routing Dashboard - Data Engine")
    print("=" * 70)

    # 1. Fetch
    baskets_raw = fetch_litter_baskets()
    complaints_raw = fetch_311_complaints()

    # 2. Clean
    print("-" * 70)
    print("[clean] Cleaning litter basket coordinates ...")
    baskets_clean = clean_coordinates(baskets_raw)

    print("[clean] Cleaning 311 complaint coordinates ...")
    complaints_clean = clean_coordinates(complaints_raw)

    if baskets_clean.empty or complaints_clean.empty:
        print(
            "[error] One of the cleaned datasets is empty; aborting "
            "enrichment stage."
        )
        return None

    # 3. Enrich
    print("-" * 70)
    print("[enrich] Starting spatial feature engineering ...")
    baskets_enriched = enrich_litter_baskets(baskets_clean, complaints_clean)

    # 4. Save
    print("-" * 70)
    save_outputs(baskets_enriched, complaints_clean)

    # Summary
    print("-" * 70)
    print("[summary] Pipeline complete.")
    print(f"[summary] Litter baskets processed : {len(baskets_enriched)}")
    print(f"[summary] 311 complaints processed  : {len(complaints_clean)}")
    print(
        f"[summary] Bins flagged as school zone: "
        f"{int(baskets_enriched['is_school_zone'].sum())}"
    )
    print(
        "[summary] Avg dist to nearest complaint (m): "
        f"{baskets_enriched['dist_to_nearest_complaint_m'].mean():.1f}"
    )
    print("=" * 70)

    return baskets_enriched


def main() -> None:
    try:
        run_pipeline()
    except requests.exceptions.RequestException as exc:
        print(f"[error] Network/API request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyError as exc:
        print(f"[error] Data schema issue: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net for CLI use
        print(f"[error] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
