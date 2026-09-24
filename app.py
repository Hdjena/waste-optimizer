"""Waste-collection dispatch dashboard.

Run:  pip install streamlit streamlit-folium folium pydeck altair numpy pandas
      streamlit run app.py
All bin data is synthetic (seeded); swap `load_bins()` for your telemetry feed.
"""
import base64
import math
from datetime import datetime, timedelta

import altair as alt
import folium
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium

st.set_page_config(page_title="Dispatch Console", page_icon="🚛", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetric"]{background: rgba(128, 128, 128, 0.1); border-left:5px solid #A94A4D; padding:.7rem 1rem; border-radius:4px}
    [data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
    </style>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ assumptions
CENTER = (30.2672, -97.7431)
DEPOT = (CENTER[0] - 0.05, CENTER[1] - 0.05)
N_BINS, STOPS_PER_TRUCK = 64, 14
MPH, STOP_MIN, START_MIN = 15.0, 3.0, 390  # avg speed, min per stop, 06:30 start
WINDOWS = [(450, 540), (840, 960)]  # school drop-off/pick-up: 07:30-09:00, 14:00-16:00
WORK_DAYS = 250
DIESEL = dict(energy=3.90 / 3.0, maint=0.85, co2=10.21 / 3.0)  # $/mi, $/mi, kg/mi
EV = dict(energy=2.6 * 0.16, maint=0.55, co2=2.6 * 0.37)
LABOR = 70.0  # $/hr, 2-person crew
TRUCK_ICON = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="44" height="28"><rect x="1" y="3" width="26" '
    b'height="16" rx="2" fill="#f59e0b" stroke="#222"/><path d="M27 8h9l6 6v5H27z" fill="#2563eb" '
    b'stroke="#222"/><circle cx="10" cy="22" r="4" fill="#222"/><circle cx="34" cy="22" r="4" fill="#222"/></svg>'
).decode()


# ------------------------------------------------------------------ data + routing
@st.cache_data
def load_bins() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "lat": CENTER[0] + rng.normal(0, 0.028, N_BINS),
        "lon": CENTER[1] + rng.normal(0, 0.028, N_BINS),
        "fill": np.clip(rng.beta(2.2, 1.6, N_BINS) * 100, 5, 100).round(),
        "school": False,
    })
    df.loc[rng.choice(N_BINS, 9, replace=False), "school"] = True
    df.index.name = "id"
    return df


BINS = load_bins()
POS = {i: (r.lat, r.lon) for i, r in BINS.iterrows()}


def miles(a, b):  # street-grid (Manhattan) distance
    return abs(a[0] - b[0]) * 69.0 + abs(a[1] - b[1]) * 69.0 * math.cos(math.radians(a[0]))


def in_window(t):
    return any(s <= t < e for s, e in WINDOWS)


def nn_route(ids, school_aware):
    """Nearest-neighbour ordering; optionally avoids arriving at school bins inside drop-off windows."""
    left, pos, clock, seq = set(ids), DEPOT, START_MIN, []
    while left:
        d = {i: miles(pos, POS[i]) for i in left}
        ok = [i for i in left if not (school_aware and BINS.school[i] and in_window(clock + d[i] / MPH * 60))]
        nxt = min(ok or left, key=d.get)
        clock += d[nxt] / MPH * 60 + STOP_MIN
        seq.append(nxt)
        pos = POS[nxt]
        left.remove(nxt)
    return seq


def plan(ids, school_aware):
    """Sweep-partition stops around the depot into trucks, then order each truck's stops."""
    ids = sorted(ids, key=lambda i: math.atan2(POS[i][0] - DEPOT[0], POS[i][1] - DEPOT[1]))
    k = max(1, math.ceil(len(ids) / STOPS_PER_TRUCK))
    return {t: nn_route(list(c), school_aware) for t, c in enumerate(np.array_split(ids, k)) if len(c)}


@st.cache_data
def baseline_routes():
    return plan(list(BINS.index), school_aware=False)


def path_points(seq):
    """Fetch actual road routes via OSRM API, fallback to direct lines."""
    import requests
    
    pts = [DEPOT] + [POS[i] for i in seq] + [DEPOT]
    
    # The routing API needs coordinates formatted as longitude,latitude
    coord_str = ";".join([f"{p[1]},{p[0]}" for p in pts])
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    
    try:
        # Ask the API for the exact road geometry
        res = requests.get(url, timeout=5).json()
        coords = res["routes"][0]["geometry"]["coordinates"]
        # Convert back to latitude,longitude for our map
        return [[lat, lon] for lon, lat in coords]
    except Exception:
        # Safe fallback: draw direct straight lines if the API is busy
        return [[p[0], p[1]] for p in pts]

def schedule(routes) -> pd.DataFrame:
    rows = []
    for t, seq in routes.items():
        clock, pos = START_MIN, DEPOT
        for n, i in enumerate(seq, 1):
            clock += miles(pos, POS[i]) / MPH * 60
            rows.append(dict(truck=t, seq=n, id=i, eta=clock, fill=BINS.fill[i], school=bool(BINS.school[i]),
                             conflict=bool(BINS.school[i]) and in_window(clock)))
            clock += STOP_MIN
            pos = POS[i]
    return pd.DataFrame(rows, columns=["truck", "seq", "id", "eta", "fill", "school", "conflict"])


def totals(routes):
    mi = sum(miles(a, b) for s in routes.values() for a, b in zip([DEPOT] + [POS[i] for i in s], [POS[i] for i in s] + [DEPOT]) if s)
    stops = sum(len(s) for s in routes.values())
    return mi, mi / MPH + stops * STOP_MIN / 60  # miles, labor hours


def cost(mi, hrs, ev_share):
    mix = lambda k: mi * (ev_share * EV[k] + (1 - ev_share) * DIESEL[k])
    return dict(Energy=mix("energy"), Maintenance=mix("maint"), Labor=hrs * LABOR), mix("co2")


def hhmm(m):
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


# ------------------------------------------------------------------ state
def regenerate():
    thr = st.session_state.thr
    st.session_state.routes = plan(BINS.index[BINS.fill >= thr], school_aware=True)
    st.session_state.plan_thr = thr
    for k in [k for k in st.session_state if k.startswith("sel_")]:
        del st.session_state[k]


if "routes" not in st.session_state:
    st.session_state.thr = 75
    regenerate()


def move(t, i, d):
    r = st.session_state.routes[t]
    a = r.index(i)
    if 0 <= a + d < len(r):
        r[a], r[a + d] = r[a + d], r[a]


def remove(t, i):
    st.session_state.routes[t].remove(i)
    st.session_state.pop(f"sel_stop_{t}", None)


def add(t, i):
    st.session_state.routes[t].append(i)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Dispatch controls")
    st.slider("Dispatch Fill Threshold", 50, 90, 75, format="%d%%", key="thr")
    ev_mix = st.slider("EV Fleet Mix (%)", 0, 100, 30, step=5, format="%d%%") / 100
    if st.button("🚀 Generate Today's Optimized Routes", type="primary", use_container_width=True):
        regenerate()
        st.toast("✅ Routes successfully optimized!", icon="🚛")
    if st.session_state.thr != st.session_state.plan_thr:
        st.warning("Threshold changed. Generate routes to apply it.")
    st.caption(f"Routes planned at {st.session_state.plan_thr}% threshold. Bins are synthetic demo data.")

routes = st.session_state.routes
served = {i for s in routes.values() for i in s}
base_mi, base_hr = totals(baseline_routes())
opt_mi, opt_hr = totals(routes)
base_cost, base_co2 = cost(base_mi, base_hr, 0.0)
opt_cost, opt_co2 = cost(opt_mi, opt_hr, ev_mix)
no_ev_cost, _ = cost(opt_mi, opt_hr, 0.0)
saved = sum(base_cost.values()) - sum(opt_cost.values())
ev_part = sum(no_ev_cost.values()) - sum(opt_cost.values())

# ------------------------------------------------------------------ metrics
st.title("Dispatch console")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bins serviced / skipped", f"{len(served)} / {N_BINS - len(served)}",
          f"{(N_BINS - len(served)) / N_BINS:.0%} of bins skipped", delta_color="off")
c2.metric("Miles saved vs baseline", f"{base_mi - opt_mi:,.1f} mi", f"{(base_mi - opt_mi) / base_mi:.0%} shorter")
c3.metric("Operational cost saved", f"${saved:,.0f}", f"+${ev_part:,.0f} from {ev_mix:.0%} EV mix")
c4.metric("CO₂ emissions saved", f"{base_co2 - opt_co2:,.0f} kg", f"{(base_co2 - opt_co2) / base_co2:.0%} lower")

tab1, tab2, tab3 = st.tabs(["Live Dispatch Map", "Manual Route Override", "EV & Financial Scenario Planner"])

# ------------------------------------------------------------------ tab 1: folium map
with tab1:
    sim_col, speed_col = st.columns([1, 2])
    simulate = sim_col.toggle("▶️ Simulate Live Truck Dispatch")
    speed = speed_col.slider("Playback speed", 1, 10, 5, disabled=not simulate)

    m = folium.Map(location=CENTER, zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(DEPOT, tooltip="Depot", icon=folium.Icon(color="black", icon="home")).add_to(m)
    shades = ["#1d4ed8", "#2563eb", "#0ea5e9", "#3b82f6", "#1e3a8a"]
    for t, seq in routes.items():
        if seq:
            folium.PolyLine(path_points(seq), color=shades[t % 5], weight=4, opacity=.85,
                            tooltip=f"Truck {t + 1}: {len(seq)} stops").add_to(m)
    for i, r in BINS.iterrows():
        on = i in served
        color = "#facc15" if r.school else ("#dc2626" if on else "#16a34a")
        folium.CircleMarker(
            POS[i], radius=8 if on else 5, color="#222" if r.school else color, weight=1.5,
            fill=True, fill_color=color, fill_opacity=.9,
            tooltip=f"BIN-{i:03d} · {r.fill:.0f}% · {'serviced' if on else 'skipped'}" + (" · school zone" if r.school else ""),
        ).add_to(m)
 
      m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;color:#000;padding:8px 12px;'
        'border:1px solid #bbb;border-radius:4px;font:12px sans-serif;line-height:1.6">'
        '<span style="color:#dc2626">●</span> Above threshold (serviced)<br>'
        '<span style="color:#16a34a">●</span> Skipped<br>'
        '<span style="color:#facc15">●</span> School-zone bin<br>'
        '<span style="color:#2563eb">━</span> Truck path</div>'))

    if simulate:  # client-side animation: smooth, no rerun loop needed
        paths = {t: np.array(path_points(s)) for t, s in routes.items() if s}
        lens = {t: np.r_[0, np.cumsum(np.abs(np.diff(p, axis=0)).sum(1))] for t, p in paths.items()}
        longest, frames, t0, feats = max(l[-1] for l in lens.values()), 260, datetime(2026, 1, 1, 6, 30), []
        for t, p in paths.items():
            n = max(2, int(frames * lens[t][-1] / longest))
            d = np.linspace(0, lens[t][-1], n)
            lat, lon = np.interp(d, lens[t], p[:, 0]), np.interp(d, lens[t], p[:, 1])
            for f in range(frames):
                j = min(f, n - 1)
                feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(lon[j]), float(lat[j])]},
                              "properties": {"time": (t0 + timedelta(seconds=f)).isoformat(), "icon": "marker",
                                             "iconstyle": {"iconUrl": TRUCK_ICON, "iconSize": [44, 28], "iconAnchor": [22, 14]},
                                             "popup": f"Truck {t + 1}"}})
        TimestampedGeoJson({"type": "FeatureCollection", "features": feats}, period="PT1S", duration="PT1S",
                           transition_time=int(500 / speed), auto_play=True, loop=True, add_last_point=False,
                           date_options="HH:mm:ss").add_to(m)

    sig = hash((tuple((t, tuple(s)) for t, s in routes.items()), simulate, speed))
    st_folium(m, height=580, use_container_width=True, returned_objects=[], key=f"map_{sig}")

# ------------------------------------------------------------------ tab 2: manual override
with tab2:
    st.caption("Edits apply to the map, the metric cards, and the school-zone checks immediately.")
    left, right = st.columns([1, 1.2])
    with left:
        truck = st.selectbox("Truck", list(routes), format_func=lambda t: f"Truck {t + 1}", key="sel_truck")
        seq = routes[truck]
        sched = schedule({truck: seq})
        if sched.empty:
            st.info("This truck has no stops. Add a skipped bin below.")
        else:
            show = pd.DataFrame({"#": sched.seq, "Bin": [f"BIN-{i:03d}" for i in sched.id], "Fill %": sched.fill,
                                 "ETA": sched.eta.map(hhmm),
                                 "School": np.where(sched.conflict, "⚠ drop-off window", np.where(sched.school, "school zone", ""))})
            st.dataframe(show, hide_index=True, use_container_width=True, height=300)
            sel = st.selectbox("Stop to change", seq, format_func=lambda i: f"BIN-{i:03d}", key=f"sel_stop_{truck}")
            b1, b2, b3 = st.columns(3)
            b1.button("⬆ Move up", on_click=move, args=(truck, sel, -1), use_container_width=True)
            b2.button("⬇ Move down", on_click=move, args=(truck, sel, 1), use_container_width=True)
            b3.button("🗑 Remove", on_click=remove, args=(truck, sel), use_container_width=True)
        skipped = [i for i in BINS.index if i not in served]
        if skipped:
            pick = st.selectbox("Add a skipped bin to this truck", skipped, key="sel_add",
                                format_func=lambda i: f"BIN-{i:03d} ({BINS.fill[i]:.0f}%)")
            st.button("➕ Add to end of route", on_click=add, args=(truck, pick))
        st.button("↺ Reset to optimized plan", on_click=regenerate)
    with right:
        layers = [pdk.Layer("PathLayer", [dict(path=[[p[1], p[0]] for p in path_points(s)], name=f"Truck {t + 1}")
                                          for t, s in routes.items() if s],
                            get_path="path", get_color=[31, 78, 121], width_min_pixels=4, pickable=True)]
        pts = pd.DataFrame([dict(lat=DEPOT[0], lon=DEPOT[1], label="Depot", color=[20, 20, 20])] + [
            dict(lat=POS[i][0], lon=POS[i][1], label=f"BIN-{i:03d}", color=[250, 204, 21] if BINS.school[i] else [220, 38, 38])
            for i in served])
        layers.append(pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"], get_fill_color="color",
                                get_radius=110, pickable=True))
        st.pydeck_chart(pdk.Deck(layers=layers, map_style="light", tooltip={"text": "{label}{name}"},
                                 initial_view_state=pdk.ViewState(latitude=CENTER[0], longitude=CENTER[1], zoom=11, pitch=0)))
        st.caption(f"Current plan: {opt_mi:,.1f} mi across {len(routes)} trucks ({opt_mi - totals(plan(BINS.index[BINS.fill >= st.session_state.plan_thr], True))[0]:+.1f} mi vs. optimizer).")

# ------------------------------------------------------------------ tab 3: scenarios
with tab3:
    a, b, c = st.columns(3)
    fleet = a.number_input("Fleet size to model (trucks)", 1, 500, 40)
    premium = b.number_input("EV price premium per truck ($)", 0, 500_000, 220_000, step=10_000)
    incentive = c.number_input("Incentive per EV truck ($)", 0, 300_000, 80_000, step=5_000)

    scale = fleet / max(1, len(routes)) * WORK_DAYS
    scen = {"Diesel fleet": 0.0, "EV fleet": 1.0, f"Current mix ({ev_mix:.0%} EV)": ev_mix}
    annual, rows, cum = {}, [], []
    for name, share in scen.items():
        parts, _ = cost(opt_mi, opt_hr, share)
        annual[name] = sum(parts.values()) * scale
        rows += [dict(Scenario=name, Category=k, Cost=v * scale) for k, v in parts.items()]
        capex = round(fleet * share) * (premium - incentive)
        cum += [dict(Month=m, Scenario=name, Cost=capex + annual[name] * m / 12) for m in range(0, 13)]
    names = list(scen)
    net_gain = annual[names[0]] - annual[names[2]]
    capex_mix = round(fleet * ev_mix) * (premium - incentive)
    be = next((r["Month"] for r in cum if r["Scenario"] == names[2] and r["Month"] > 0 and
               r["Cost"] <= next(x["Cost"] for x in cum if x["Scenario"] == names[0] and x["Month"] == r["Month"])), None)
    m1, m2, m3 = st.columns(3)
    m1.metric("Annual operating savings (current mix)", f"${net_gain:,.0f}")
    m2.metric("Upfront net EV investment", f"${capex_mix:,.0f}")
    m3.metric("Payback within 12 months", f"Month {be}" if be else "Not within 1 year")

    ch1, ch2 = st.columns(2)
    ch1.subheader("Cumulative cost over 1 year")
    ch1.altair_chart(alt.Chart(pd.DataFrame(cum)).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Month:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("Cost:Q", title="Cumulative cost ($)"),
        color=alt.Color("Scenario:N", legend=alt.Legend(orient="bottom")), tooltip=["Scenario", "Month", alt.Tooltip("Cost:Q", format=",.0f")]),
        use_container_width=True)
    ch2.subheader("Annual operating cost by category")
    ch2.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("Scenario:N", sort=names, axis=alt.Axis(labelAngle=0), title=None), y=alt.Y("sum(Cost):Q", title="$ per year"),
        color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")), tooltip=["Scenario", "Category", alt.Tooltip("Cost:Q", format=",.0f")]),
        use_container_width=True)

    st.subheader("School drop-off zone compliance")
    st.caption("No arrivals at school-zone bins during 07:30–09:00 or 14:00–16:00.")
    s_now, s_base = schedule(routes), schedule(baseline_routes())
    s_now, s_base = s_now[s_now.school], s_base[s_base.school]
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("School-zone stops", len(s_now))
    z2.metric("Window conflicts", int(s_now.conflict.sum()), f"{int(s_now.conflict.sum() - s_base.conflict.sum()):+d} vs. baseline", delta_color="inverse")
    z3.metric("Compliance", f"{1 - s_now.conflict.mean():.0%}" if len(s_now) else "n/a",
              f"baseline {1 - s_base.conflict.mean():.0%}", delta_color="off")
    z4.metric("Zero-emission school stops", f"{round(len(s_now) * ev_mix)} of {len(s_now)}")
    if len(s_now):
        day = datetime(2026, 1, 1)
        sdf = pd.DataFrame({"Stop": [f"BIN-{i:03d}" for i in s_now.id], "Arrival": [day + timedelta(minutes=float(e)) for e in s_now.eta],
                            "Status": np.where(s_now.conflict, "In drop-off window", "Compliant")})
        win = pd.DataFrame([dict(s=day + timedelta(minutes=s), e=day + timedelta(minutes=e)) for s, e in WINDOWS])
        shade = alt.Chart(win).mark_rect(opacity=.2, color="#dc2626").encode(x="s:T", x2="e:T")
        dots = alt.Chart(sdf).mark_point(filled=True, size=150).encode(
            x=alt.X("Arrival:T", axis=alt.Axis(format="%H:%M"), scale=alt.Scale(domain=[day + timedelta(hours=6), day + timedelta(hours=17)])),
            y=alt.Y("Stop:N", title=None),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Compliant", "In drop-off window"], range=["#16a34a", "#dc2626"])),
            tooltip=["Stop", alt.Tooltip("Arrival:T", format="%H:%M"), "Status"])
        st.altair_chart(shade + dots, use_container_width=True)
    else:
        st.info("No school-zone bins are scheduled at this threshold.")
