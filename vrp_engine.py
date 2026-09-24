"""
vrp_engine.py
=============

Predictive fill-level scoring + Vehicle Routing Problem with Time Windows
(VRPTW) solver for a dynamic NYC municipal waste routing dashboard.

Designed to consume the enriched litter-basket dataset produced by
`data_engine.py` (data/litter_baskets_enriched.csv), which must contain:
    latitude, longitude,
    dist_to_nearest_complaint_m, complaints_within_300m,
    dist_to_nearest_school_m, is_school_zone

Pipeline stages
----------------
1. predict_fill_levels   -> RandomForestRegressor estimates each bin's
                             current fill percentage from spatial features.
2. solve_vrptw            -> OR-Tools VRPTW solve over bins whose predicted
                             fill exceeds a threshold, with:
                               - 50-bin truck capacity
                               - drop penalties scaled by 311 hotspot density
                               - school-zone pickup windows that exclude the
                                 7:00-9:30 AM drop-off window
3. calculate_fleet_impact -> cost/CO2 comparison of the solved route under a
                             mixed EV/diesel fleet vs. a 100% diesel baseline.

Run directly:
    python vrp_engine.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_CSV = os.path.join("data", "litter_baskets_enriched.csv")
SCORED_OUTFILE = os.path.join("data", "vrp_scored_bins.csv")
ROUTES_OUTFILE = os.path.join("data", "vrp_routes.csv")

FEATURE_COLS = [
    "dist_to_nearest_complaint_m",
    "complaints_within_300m",
    "dist_to_nearest_school_m",
]

EARTH_RADIUS_M = 6_371_000.0

# Fleet / VRP constants
VEHICLE_CAPACITY_BINS = 50
FILL_THRESHOLD_DEFAULT = 60.0
AVG_TRUCK_SPEED_KMH = 25.0
SERVICE_TIME_MIN = 3
SOLVER_TIME_LIMIT_S = 20

# Time horizon in minutes (0 = midnight, 1440 = end of day)
DAY_START_MIN = 0
DAY_END_MIN = 1440
SCHOOL_DROPOFF_START_MIN = 420   # 7:00 AM
SCHOOL_DROPOFF_END_MIN = 570     # 9:30 AM

# 311 hotspot drop-penalty scaling: penalty = BASE + WEIGHT * complaints_within_300m
HOTSPOT_PENALTY_BASE = 2_000
HOTSPOT_PENALTY_WEIGHT = 500
SCHOOL_PAIR_PENALTY = 50_000  # strongly discourage skipping a school-zone bin entirely

# EV vs Diesel constants
DIESEL_MPG = 3.0
DIESEL_PRICE_PER_GAL = 5.00
DIESEL_CO2_KG_PER_MILE = 1.47

EV_KWH_PER_MILE = 1.2
EV_PRICE_PER_KWH = 0.18
EV_CO2_KG_PER_MILE = 0.20

METERS_PER_MILE = 1609.344


# --------------------------------------------------------------------------
# 1. Predictive Fill Model
# --------------------------------------------------------------------------

def predict_fill_levels(
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    noise_std: float = 8.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a RandomForestRegressor to estimate `predicted_fill_percentage`
    (0-100) for each bin from spatial features.

    NOTE: NYC Open Data does not expose live IoT bin-sensor fill readings,
    so this function simulates a plausible ground-truth fill level as a
    function of the spatial features plus Gaussian noise, then trains/
    evaluates the RandomForest against that synthetic target. This keeps
    the full ML pipeline (train/test split, fit, evaluate, predict)
    realistic and swappable for real sensor data later -- only the
    synthetic-target block below would need to be replaced.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    out = df.copy()
    rng = np.random.default_rng(random_state)

    X = out[feature_cols].fillna(out[feature_cols].median())

    # --- Synthetic ground-truth fill level (demo target) ---
    # Bins closer to / surrounded by more 311 sanitation complaints tend to
    # fill up faster (foot traffic, dumping); bins far from schools/complaint
    # clusters are assumed to fill more slowly.
    synthetic_truth = (
        45.0
        + 0.08 * X["complaints_within_300m"].to_numpy()
        - 0.01 * X["dist_to_nearest_complaint_m"].to_numpy()
        - 0.004 * X["dist_to_nearest_school_m"].to_numpy()
        + rng.normal(0, noise_std, size=len(X))
    )
    synthetic_truth = np.clip(synthetic_truth, 0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, synthetic_truth, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(
        f"[predict] RandomForestRegressor trained on {len(X_train)} rows "
        f"(holdout MAE={mae:.2f} pts, R2={r2:.3f})."
    )

    predicted = np.clip(model.predict(X), 0, 100)
    out["predicted_fill_percentage"] = predicted.round(1)
    print(
        "[predict] Scored "
        f"{len(out)} bins. Mean predicted fill: "
        f"{out['predicted_fill_percentage'].mean():.1f}%."
    )
    return out


# --------------------------------------------------------------------------
# Spatial helpers
# --------------------------------------------------------------------------

def haversine_distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized pairwise haversine distance matrix (meters)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dlat = lat_rad[None, :] - lat_rad[:, None]
    dlon = lon_rad[None, :] - lon_rad[:, None]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_M * c


def ensure_bin_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a stable `bin_id` column, falling back to the row index."""
    out = df.copy()
    if "bin_id" in out.columns:
        return out
    for candidate in ("objectid", "the_geom", "unique_id", "asset_id"):
        if candidate in out.columns:
            out["bin_id"] = out[candidate].astype(str)
            return out
    out = out.reset_index(drop=False).rename(columns={"index": "bin_id"})
    out["bin_id"] = "BIN-" + out["bin_id"].astype(str)
    return out


def _coerce_school_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure is_school_zone is a real bool (CSV round-trips as strings)."""
    out = df.copy()
    if out["is_school_zone"].dtype != bool:
        out["is_school_zone"] = (
            out["is_school_zone"].astype(str).str.strip().str.lower().eq("true")
        )
    return out


# --------------------------------------------------------------------------
# 2. Advanced OR-Tools VRPTW Engine
# --------------------------------------------------------------------------

def _prepare_vrp_nodes(filtered_df: pd.DataFrame):
    """
    Build per-node arrays for the solver. School-zone bins are expanded into
    two candidate nodes (early / late pickup window) linked by a disjunction
    so the solver picks exactly one; non-school bins get a single node with
    an all-day window.

    Returns
    -------
    lats, lons, demands, tw_start, tw_end, service : np.ndarray
    bin_ids : list (one entry per node, aligned with the arrays above)
    school_pairs : list[tuple[int, int]] of node-array indices (0-based,
                   depot excluded) representing the same physical bin
    """
    lats, lons, demands, tw_start, tw_end, service, bin_ids = [], [], [], [], [], [], []
    school_pairs = []

    for _, row in filtered_df.iterrows():
        if bool(row["is_school_zone"]):
            pair_start = len(lats)
            windows = [
                (DAY_START_MIN, SCHOOL_DROPOFF_START_MIN),
                (SCHOOL_DROPOFF_END_MIN, DAY_END_MIN),
            ]
            for start, end in windows:
                lats.append(row["latitude"])
                lons.append(row["longitude"])
                demands.append(1)
                tw_start.append(start)
                tw_end.append(end)
                service.append(SERVICE_TIME_MIN)
                bin_ids.append(row["bin_id"])
            school_pairs.append((pair_start, pair_start + 1))
        else:
            lats.append(row["latitude"])
            lons.append(row["longitude"])
            demands.append(1)
            tw_start.append(DAY_START_MIN)
            tw_end.append(DAY_END_MIN)
            service.append(SERVICE_TIME_MIN)
            bin_ids.append(row["bin_id"])

    return (
        np.array(lats, dtype=float),
        np.array(lons, dtype=float),
        np.array(demands, dtype=int),
        np.array(tw_start, dtype=int),
        np.array(tw_end, dtype=int),
        np.array(service, dtype=int),
        bin_ids,
        school_pairs,
    )


def solve_vrptw(
    scored_df: pd.DataFrame,
    threshold: float = FILL_THRESHOLD_DEFAULT,
    vehicle_capacity: int = VEHICLE_CAPACITY_BINS,
    avg_speed_kmh: float = AVG_TRUCK_SPEED_KMH,
    num_vehicles: Optional[int] = None,
) -> dict:
    """
    Filter bins by predicted fill %, then solve a VRPTW where the depot is
    the centroid of the filtered bins.

    Returns a dict with the solved routes, dropped bins, total distance,
    and solver metadata. Returns {"status": "no_bins_above_threshold"} or
    {"status": "infeasible"} if no solution could be produced.
    """
    if not ORTOOLS_AVAILABLE:
        raise ImportError(
            "OR-Tools is not installed. Run `pip install ortools --break-system-packages`."
        )

    candidates = scored_df[scored_df["predicted_fill_percentage"] > threshold].copy()
    candidates = candidates.reset_index(drop=True)
    print(
        f"[vrp] {len(candidates)}/{len(scored_df)} bins exceed the "
        f"{threshold:.0f}% fill threshold and are eligible for pickup."
    )
    if candidates.empty:
        return {"status": "no_bins_above_threshold"}

    depot_lat = candidates["latitude"].mean()
    depot_lon = candidates["longitude"].mean()
    print(f"[vrp] Depot set to centroid ({depot_lat:.5f}, {depot_lon:.5f}).")

    (lats, lons, demands, tw_start, tw_end, service,
     bin_ids, school_pairs) = _prepare_vrp_nodes(candidates)

    # Prepend depot as node 0
    all_lats = np.concatenate([[depot_lat], lats])
    all_lons = np.concatenate([[depot_lon], lons])
    all_demands = np.concatenate([[0], demands])
    all_tw_start = np.concatenate([[DAY_START_MIN], tw_start])
    all_tw_end = np.concatenate([[DAY_END_MIN], tw_end])
    all_service = np.concatenate([[0], service])
    all_bin_ids = ["DEPOT"] + bin_ids
    # school_pairs indices were relative to the node arrays (0-based, no
    # depot); shift by +1 to account for depot at index 0.
    school_pairs = [(a + 1, b + 1) for a, b in school_pairs]

    num_nodes = len(all_lats)
    distance_matrix_m = haversine_distance_matrix(all_lats, all_lons)

    if num_vehicles is None:
        num_vehicles = max(1, math.ceil(len(candidates) / vehicle_capacity) + 1)
    print(f"[vrp] Fleet size: {num_vehicles} trucks (capacity {vehicle_capacity} bins each).")

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- Distance (arc cost) callback ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(distance_matrix_m[from_node][to_node]))

    distance_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_cb_idx)

    # --- Capacity dimension ---
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(all_demands[from_node])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [vehicle_capacity] * num_vehicles, True, "Capacity"
    )

    # --- Time dimension (travel time + service time), minutes ---
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_min = distance_matrix_m[from_node][to_node] / 1000.0 / avg_speed_kmh * 60.0
        return int(round(travel_min + all_service[from_node]))

    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_cb_idx, DAY_END_MIN, DAY_END_MIN, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for node in range(num_nodes):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(
            int(all_tw_start[node]), int(all_tw_end[node])
        )

    for vehicle_id in range(num_vehicles):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        time_dimension.CumulVar(start_index).SetRange(DAY_START_MIN, DAY_END_MIN)
        time_dimension.CumulVar(end_index).SetRange(DAY_START_MIN, DAY_END_MIN)

    # --- Drop penalties: 311 hotspot bins get a high penalty so the
    #     solver strongly prefers to keep them on a route (prioritized). ---
    complaint_density = candidates["complaints_within_300m"].to_numpy()
    node_to_bin_row = {}  # node index -> row index in `candidates`
    node_cursor = 1
    for row_i in range(len(candidates)):
        if bool(candidates.loc[row_i, "is_school_zone"]):
            node_to_bin_row[node_cursor] = row_i
            node_to_bin_row[node_cursor + 1] = row_i
            node_cursor += 2
        else:
            node_to_bin_row[node_cursor] = row_i
            node_cursor += 1

    handled_nodes = set()
    for a, b in school_pairs:
        row_i = node_to_bin_row[a]
        hotspot_penalty = int(
            HOTSPOT_PENALTY_BASE + HOTSPOT_PENALTY_WEIGHT * complaint_density[row_i]
        )
        pair_penalty = max(SCHOOL_PAIR_PENALTY, hotspot_penalty)
        routing.AddDisjunction(
            [manager.NodeToIndex(a), manager.NodeToIndex(b)], pair_penalty, 1
        )
        handled_nodes.update([a, b])

    for node in range(1, num_nodes):
        if node in handled_nodes:
            continue
        row_i = node_to_bin_row[node]
        hotspot_penalty = int(
            HOTSPOT_PENALTY_BASE + HOTSPOT_PENALTY_WEIGHT * complaint_density[row_i]
        )
        routing.AddDisjunction([manager.NodeToIndex(node)], hotspot_penalty)

    # --- Solve ---
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(SOLVER_TIME_LIMIT_S)

    print("[vrp] Solving VRPTW ...")
    solution = routing.SolveWithParameters(search_params)

    if solution is None:
        print("[vrp] No feasible solution found within the time limit.")
        return {"status": "infeasible"}

    routes = []
    total_distance_m = 0.0
    visited_nodes = set()

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        stops = []
        route_distance_m = 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                visited_nodes.add(node)
                arrival = solution.Value(time_dimension.CumulVar(index))
                stops.append(
                    {
                        "bin_id": all_bin_ids[node],
                        "latitude": round(float(all_lats[node]), 6),
                        "longitude": round(float(all_lons[node]), 6),
                        "arrival_time_min": int(arrival),
                    }
                )
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            route_distance_m += distance_matrix_m[node][next_node]
        if stops:
            routes.append(
                {
                    "vehicle_id": vehicle_id,
                    "stops": stops,
                    "num_bins": len(stops),
                    "route_distance_m": round(route_distance_m, 1),
                }
            )
            total_distance_m += route_distance_m

    dropped_nodes = [n for n in range(1, num_nodes) if n not in visited_nodes]
    # De-duplicate school-zone pairs so a dropped bin isn't reported twice.
    dropped_bin_ids = sorted({all_bin_ids[n] for n in dropped_nodes})
    served_bin_ids = {all_bin_ids[n] for n in visited_nodes}
    dropped_bin_ids = [b for b in dropped_bin_ids if b not in served_bin_ids]

    total_distance_miles = total_distance_m / METERS_PER_MILE

    print(
        f"[vrp] Solved: {len(routes)} routes used, "
        f"{len(visited_nodes)} bin-visits, {len(dropped_bin_ids)} bins dropped, "
        f"{total_distance_miles:.1f} total route miles."
    )

    return {
        "status": "solved",
        "depot": (depot_lat, depot_lon),
        "routes": routes,
        "dropped_bins": dropped_bin_ids,
        "total_distance_m": total_distance_m,
        "total_distance_miles": total_distance_miles,
        "num_vehicles_used": len(routes),
    }


# --------------------------------------------------------------------------
# 3. EV vs. Diesel Financial / Emissions Calculator
# --------------------------------------------------------------------------

def calculate_fleet_impact(total_miles: float, ev_fleet_percentage: float) -> dict:
    """
    Compare a mixed EV/diesel fleet driving `total_miles` against a
    100%-diesel baseline covering the same mileage.

    Parameters
    ----------
    total_miles : float
        Total route miles driven by the solved fleet (all vehicles combined).
    ev_fleet_percentage : float
        Share (0-100) of those miles driven by EV trucks; the remainder is
        assumed diesel.

    Returns
    -------
    dict with total cost ($), total CO2 (kg), fuel dollars saved, and CO2
    saved vs. the 100%-diesel baseline.
    """
    ev_pct = max(0.0, min(100.0, ev_fleet_percentage)) / 100.0
    diesel_miles = total_miles * (1 - ev_pct)
    ev_miles = total_miles * ev_pct

    diesel_cost = (diesel_miles / DIESEL_MPG) * DIESEL_PRICE_PER_GAL
    diesel_co2_kg = diesel_miles * DIESEL_CO2_KG_PER_MILE

    ev_cost = ev_miles * EV_KWH_PER_MILE * EV_PRICE_PER_KWH
    ev_co2_kg = ev_miles * EV_CO2_KG_PER_MILE

    total_cost = diesel_cost + ev_cost
    total_co2_kg = diesel_co2_kg + ev_co2_kg

    baseline_cost = (total_miles / DIESEL_MPG) * DIESEL_PRICE_PER_GAL
    baseline_co2_kg = total_miles * DIESEL_CO2_KG_PER_MILE

    return {
        "total_miles": round(total_miles, 1),
        "ev_fleet_percentage": round(ev_fleet_percentage, 1),
        "total_cost_usd": round(total_cost, 2),
        "total_co2_kg": round(total_co2_kg, 1),
        "baseline_diesel_cost_usd": round(baseline_cost, 2),
        "baseline_diesel_co2_kg": round(baseline_co2_kg, 1),
        "fuel_dollars_saved_usd": round(baseline_cost - total_cost, 2),
        "co2_saved_kg": round(baseline_co2_kg - total_co2_kg, 1),
    }


# --------------------------------------------------------------------------
# Data loading (with a standalone demo fallback)
# --------------------------------------------------------------------------

def load_or_simulate_baskets(n_demo: int = 150, seed: int = 7) -> pd.DataFrame:
    """
    Load data/litter_baskets_enriched.csv (output of data_engine.py). If it
    is not present, simulate a small demo dataset with the same schema so
    this script can still be run standalone for demonstration purposes.
    """
    if os.path.exists(INPUT_CSV):
        print(f"[load] Reading enriched bins from {INPUT_CSV} ...")
        df = pd.read_csv(INPUT_CSV)
        print(f"[load] Loaded {len(df)} bins.")
        return df

    print(
        f"[load] WARNING: {INPUT_CSV} not found. Run data_engine.py first "
        "for real data. Falling back to a simulated demo dataset."
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "latitude": rng.uniform(40.70, 40.80, n_demo),
            "longitude": rng.uniform(-74.02, -73.93, n_demo),
            "dist_to_nearest_complaint_m": rng.uniform(10, 1200, n_demo),
            "complaints_within_300m": rng.poisson(2.5, n_demo),
            "dist_to_nearest_school_m": rng.uniform(10, 2000, n_demo),
        }
    )
    df["is_school_zone"] = df["dist_to_nearest_school_m"] <= 500
    return df


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------

def run_pipeline(
    fill_threshold: float = FILL_THRESHOLD_DEFAULT,
    ev_fleet_percentage: float = 30.0,
) -> Optional[dict]:
    print("=" * 70)
    print("NYC Waste Routing Dashboard - VRP Engine")
    print("=" * 70)

    baskets = load_or_simulate_baskets()
    baskets = ensure_bin_id(baskets)
    baskets = _coerce_school_flag(baskets)

    print("-" * 70)
    scored = predict_fill_levels(baskets)

    os.makedirs("data", exist_ok=True)
    scored.to_csv(SCORED_OUTFILE, index=False)
    print(f"[save] Wrote {len(scored)} scored bins -> {SCORED_OUTFILE}")

    print("-" * 70)
    if not ORTOOLS_AVAILABLE:
        print(
            "[vrp] OR-Tools is not installed in this environment; skipping "
            "the routing stage. Install with: "
            "pip install ortools --break-system-packages"
        )
        return None

    result = solve_vrptw(scored, threshold=fill_threshold)

    if result["status"] != "solved":
        print(f"[vrp] Routing stage ended with status: {result['status']}")
        return result

    # Flatten routes to a CSV
    rows = []
    for route in result["routes"]:
        for seq, stop in enumerate(route["stops"], start=1):
            rows.append(
                {
                    "vehicle_id": route["vehicle_id"],
                    "stop_sequence": seq,
                    **stop,
                }
            )
    routes_df = pd.DataFrame(rows)
    routes_df.to_csv(ROUTES_OUTFILE, index=False)
    print(f"[save] Wrote {len(routes_df)} route stops -> {ROUTES_OUTFILE}")

    print("-" * 70)
    impact = calculate_fleet_impact(result["total_distance_miles"], ev_fleet_percentage)
    print(
        f"[fleet] {ev_fleet_percentage:.0f}% EV fleet over "
        f"{impact['total_miles']} route miles:"
    )
    print(f"[fleet]   Total cost           : ${impact['total_cost_usd']:,.2f}")
    print(f"[fleet]   Total CO2            : {impact['total_co2_kg']:,.1f} kg")
    print(f"[fleet]   Fuel $ saved vs 100% diesel : ${impact['fuel_dollars_saved_usd']:,.2f}")
    print(f"[fleet]   CO2 saved vs 100% diesel    : {impact['co2_saved_kg']:,.1f} kg")
    print("=" * 70)

    result["fleet_impact"] = impact
    return result


def main() -> None:
    try:
        run_pipeline()
    except (KeyError, ValueError) as exc:
        print(f"[error] Data/config issue: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net for CLI use
        print(f"[error] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
