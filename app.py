# ============================================================
# URIP NAIROBI - SCENARIO-BASED URBAN INTELLIGENCE DASHBOARD
# Streamlit conversion of the final notebook dashboard
# ============================================================
#
# Dashboard workflow:
#   INCIDENT -> SCENARIO -> BEST EMERGENCY FACILITY ->
#   ALTERNATIVE ROUTES -> TRAVEL TIME + FLOOD EXPOSURE ->
#   OPERATIONAL RECOMMENDATION
#
# Required local files:
#   urip_data.pkl
#   URIP_Nairobi_Flood_Scenario.tif
#   URIP_Nairobi_Flood_Susceptibility.tif   (optional; layer is shown if present)
#
# urip_data.pkl must contain:
#   G, G_traffic, G_combined
#   hospitals_cbd, incidents_wgs84
#   flood_gdf, edges_projected
# ============================================================

import pickle
from pathlib import Path

import folium
from folium import plugins
from folium.plugins import BeautifyIcon
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import streamlit as st
from rasterio.warp import transform_bounds
from shapely.geometry import LineString
from streamlit_folium import st_folium


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="URIP Nairobi Emergency Response",
    page_icon="🚨",
    layout="wide",
)


# ------------------------------------------------------------
# DESIGN SYSTEM
# ------------------------------------------------------------

NAVY      = "#0B2545"
NAVY_2    = "#15335C"
TEAL      = "#13A9C7"
CORAL     = "#FF6B4A"
WHITE     = "#FFFFFF"
OFFWHITE  = "#F7F9FB"
TEXT_DARK = "#13294B"
MUTED     = "#5C6B7A"
CARD_BG   = "#EDF3F7"

STATUS_COLORS = {
    "Not affected":        "#2A9D8F",
    "Minor exposure":      "#F4A261",
    "Moderate disruption": "#E63946",
    "Major disruption":    "#9D0208",
    "Unknown":             MUTED,
    "Not assessed":        MUTED,
}


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

@st.cache_resource
def load_data():
    data_path = Path("urip_data.pkl")

    if not data_path.exists():
        raise FileNotFoundError(
            "urip_data.pkl was not found. Run prepare_urip_data.py in your "
            "Colab notebook after all URIP analytical cells have been run."
        )

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    required = [
        "G",
        "G_traffic",
        "G_combined",
        "hospitals_cbd",
        "incidents_wgs84",
        "flood_gdf",
        "edges_projected",
    ]

    missing = [key for key in required if key not in data]

    if missing:
        raise ValueError(
            "The data bundle is missing: " + ", ".join(missing)
        )

    return data


@st.cache_resource
def load_flood_rasters():
    flood_path = Path("URIP_Nairobi_Flood_Scenario.tif")
    susceptibility_path = Path("URIP_Nairobi_Flood_Susceptibility.tif")

    if not flood_path.exists():
        raise FileNotFoundError(
            "URIP_Nairobi_Flood_Scenario.tif was not found."
        )

    flood_raster = rasterio.open(flood_path)

    susceptibility_raster = None
    if susceptibility_path.exists():
        susceptibility_raster = rasterio.open(susceptibility_path)

    return flood_raster, susceptibility_raster


try:
    DATA = load_data()
    flood_raster, susceptibility_raster = load_flood_rasters()

    G = DATA["G"]
    G_traffic = DATA["G_traffic"]
    G_combined = DATA["G_combined"]

    hospitals_cbd = DATA["hospitals_cbd"]
    incidents_wgs84 = DATA["incidents_wgs84"]

    flood_gdf = DATA["flood_gdf"]
    edges_projected = DATA["edges_projected"]

except Exception as e:
    st.error("URIP data has not been prepared for Streamlit yet.")
    st.code(str(e))
    st.info(
        "Place urip_data.pkl and URIP_Nairobi_Flood_Scenario.tif beside "
        "app.py, then rerun the app."
    )
    st.stop()


# ------------------------------------------------------------
# PREPARE FLOOD GEOMETRY
# ------------------------------------------------------------

flood_union = flood_gdf.dissolve()
flood_geometry = flood_union.geometry.iloc[0]


# ------------------------------------------------------------
# SAFE ROUTE -> GEODATAFRAME FUNCTION
# ------------------------------------------------------------

def route_to_gdf_safe(graph, route):
    """Convert a network route into a GeoDataFrame for visualization."""

    route_edges = []

    for u, v in zip(route[:-1], route[1:]):
        try:
            edge_data = graph.get_edge_data(u, v)

            if edge_data is None:
                continue

            best_key = min(
                edge_data,
                key=lambda k: edge_data[k].get(
                    "length",
                    float("inf")
                )
            )

            data = edge_data[best_key]
            geometry = data.get("geometry")

            if geometry is None:
                node_u = graph.nodes[u]
                node_v = graph.nodes[v]

                geometry = LineString([
                    (node_u["x"], node_u["y"]),
                    (node_v["x"], node_v["y"])
                ])

            route_edges.append({
                "u": u,
                "v": v,
                "geometry": geometry
            })

        except Exception:
            continue

    if not route_edges:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(
        route_edges,
        geometry="geometry",
        crs="EPSG:4326"
    )


# ------------------------------------------------------------
# ROUTE CALCULATION
# ------------------------------------------------------------

def calculate_scenario_route(graph, origin, destination, weight):
    try:
        route = nx.shortest_path(
            graph,
            origin,
            destination,
            weight=weight
        )

        travel_time = nx.shortest_path_length(
            graph,
            origin,
            destination,
            weight=weight
        )

        return route, travel_time

    except nx.NetworkXNoPath:
        return None, np.nan


# ------------------------------------------------------------
# ALTERNATIVE ROUTES
# ------------------------------------------------------------

def get_alternative_routes(
    graph,
    origin,
    destination,
    weight,
    k=3
):
    """Generate up to k alternative routes."""

    simple_graph = nx.DiGraph()

    for u, v, key, data in graph.edges(
        keys=True,
        data=True
    ):

        edge_weight = data.get(
            weight,
            float("inf")
        )

        if not np.isfinite(edge_weight):
            continue

        if simple_graph.has_edge(u, v):

            current_weight = simple_graph[u][v].get(
                weight,
                float("inf")
            )

            if edge_weight < current_weight:
                simple_graph[u][v].update(data)

        else:
            simple_graph.add_edge(
                u,
                v,
                **data
            )

    if (
        origin not in simple_graph
        or destination not in simple_graph
    ):
        return []

    alternatives = []

    try:
        paths = nx.shortest_simple_paths(
            simple_graph,
            origin,
            destination,
            weight=weight
        )

        for path in paths:

            time_sec = nx.path_weight(
                simple_graph,
                path,
                weight=weight
            )

            alternatives.append({
                "route": path,
                "time_sec": time_sec
            })

            if len(alternatives) >= k:
                break

    except nx.NetworkXNoPath:
        return []

    return alternatives


# ------------------------------------------------------------
# FLOOD EXPOSURE
# ------------------------------------------------------------

def calculate_route_flood_exposure(
    graph,
    route,
    flood_geometry
):
    """Flood exposure = flooded route length / total route length x 100."""

    total_length = 0
    flooded_length = 0

    for u, v in zip(route[:-1], route[1:]):

        try:
            edge_data = graph.get_edge_data(u, v)

            if edge_data is None:
                continue

            best_edge = min(
                edge_data.values(),
                key=lambda x: x.get(
                    "length",
                    float("inf")
                )
            )

            geometry = best_edge.get("geometry")

            if geometry is None:

                node_u = graph.nodes[u]
                node_v = graph.nodes[v]

                geometry = LineString([
                    (node_u["x"], node_u["y"]),
                    (node_v["x"], node_v["y"])
                ])

            edge_gdf = gpd.GeoSeries(
                [geometry],
                crs="EPSG:4326"
            ).to_crs(
                flood_gdf.crs
            )

            projected_geometry = edge_gdf.iloc[0]

            flooded_segment = (
                projected_geometry
                .intersection(flood_geometry)
            )

            flooded_length += flooded_segment.length
            total_length += projected_geometry.length

        except Exception:
            continue

    if total_length == 0:
        return {
            "flooded_length_m": 0,
            "route_length_m": 0,
            "flood_percentage": 0,
            "flood_status": "Unknown"
        }

    flood_percentage = (
        flooded_length /
        total_length
    ) * 100

    if flood_percentage == 0:
        status = "Not affected"
    elif flood_percentage <= 20:
        status = "Minor exposure"
    elif flood_percentage <= 50:
        status = "Moderate disruption"
    else:
        status = "Major disruption"

    return {
        "flooded_length_m": flooded_length,
        "route_length_m": total_length,
        "flood_percentage": flood_percentage,
        "flood_status": status,
    }


# ------------------------------------------------------------
# ROUTE RECOMMENDATION (EXACT ROUTE ASSIGNMENT)
# ------------------------------------------------------------

def route_recommendation(
    routes_df,
    scenario
):
    """Explicitly names the target route number for operational guidance."""

    if routes_df.empty:
        return "NO ROUTE AVAILABLE"

    # For non-flood scenarios, explicitly specify Route 1 (fastest path)
    if scenario != "Flood + Traffic":
        best_route_num = int(routes_df.iloc[0]["route_number"])
        return f"DISPATCH VIA ROUTE {best_route_num} (FASTEST)"

    # Under Flood + Traffic, evaluate flood exposure thresholds
    viable_routes = routes_df[
        routes_df["flood_percentage"] <= 20
    ].copy()

    if not viable_routes.empty:
        best_viable = (
            viable_routes
            .sort_values("time_sec")
            .iloc[0]
        )
        route_num = int(best_viable["route_number"])

        if route_num == 1:
            return f"DISPATCH VIA ROUTE {route_num} (MINIMAL FLOOD EXPOSURE)"

        return f"REROUTE VIA ROUTE {route_num} (BYPASSES FLOODED SECTIONS)"

    return "CAUTION: ALL ALTERNATIVE ROUTES ARE HEAVILY FLOOD-EXPOSED"

# ------------------------------------------------------------
# OPERATIONAL GUIDANCE FUNCTIONS
# ------------------------------------------------------------

def get_operational_action(scenario, routes_df):
    """Tactical command action based on environmental risk level."""
    if routes_df.empty:
        return "RED ALERT: Suspend immediate ground dispatch. Initiate aerial/marine reconnaissance."

    max_flood = routes_df["flood_percentage"].max() if "flood_percentage" in routes_df.columns else 0

    if scenario == "Flood + Traffic" and max_flood > 50:
        return "HIGH RISK DISPATCH: Alert field units of active inundation. Require high-clearance response vehicles."
    elif scenario == "Peak Traffic":
        return "PRIORITY DISPATCH: Notify traffic control center to clear critical intersections along corridor."
    else:
        return "STANDARD DISPATCH: Mobilize primary unit under normal emergency response protocols."


def get_operational_recommendation(scenario, routes_df):
    """Spatial and routing recommendation naming exact target routes."""
    if routes_df.empty:
        return "NO VIABLE ROUTE IDENTIFIED"

    best_route = int(routes_df.iloc[0]["route_number"])
    
    if scenario != "Flood + Traffic":
        return f"Primary Routing: Proceed via Route {best_route} (Optimal Travel Time)"

    # Handle flood evaluation logic
    viable_routes = routes_df[routes_df["flood_percentage"] <= 20]

    if not viable_routes.empty:
        recommended_route = int(viable_routes.iloc[0]["route_number"])
        if recommended_route == 1:
            return f"Primary Routing: Proceed via Route {recommended_route} (Minimal Flood Exposure)"
        return f"Tactical Reroute: Bypass Primary Corridor; proceed via Route {recommended_route}"
    
    return f"Cautionary Routing: Proceed via Route {best_route} with extreme caution (High Flood Exposure)"


# ------------------------------------------------------------
# SCENARIO ANALYSIS
# ------------------------------------------------------------

def analyse_scenario(
    incident_id,
    scenario,
    k_routes=3
):
    """Incident -> Scenario -> Best hospital -> Alternative routes."""

    incident_row = incidents_wgs84[
        incidents_wgs84["incident_id"] == incident_id
    ]

    if incident_row.empty:
        return None

    incident_node = int(
        incident_row.iloc[0]["nearest_node"]
    )

    if scenario == "Off-Peak":
        graph = G
        weight = "travel_time"

    elif scenario == "Peak Traffic":
        graph = G_traffic
        weight = "final_traffic_time"

    elif scenario == "Flood + Traffic":
        graph = G_combined
        weight = "combined_travel_time"

    else:
        return None

    hospital_results = []

    for _, hospital in hospitals_cbd.iterrows():

        hospital_name = hospital["name"]
        hospital_node = int(
            hospital["nearest_node"]
        )

        try:

            travel_time = nx.shortest_path_length(
                graph,
                hospital_node,
                incident_node,
                weight=weight
            )

            hospital_results.append({
                "hospital": hospital_name,
                "hospital_node": hospital_node,
                "travel_time_sec": travel_time
            })

        except nx.NetworkXNoPath:
            continue

    if not hospital_results:
        return None

    hospital_df = (
        pd.DataFrame(hospital_results)
        .sort_values("travel_time_sec")
        .reset_index(drop=True)
    )

    best_hospital = hospital_df.iloc[0]

    best_hospital_name = best_hospital["hospital"]
    best_hospital_node = int(
        best_hospital["hospital_node"]
    )

    alternatives = get_alternative_routes(
        graph,
        best_hospital_node,
        incident_node,
        weight,
        k=k_routes
    )

    route_results = []

    for i, alternative in enumerate(
        alternatives,
        start=1
    ):

        route = alternative["route"]
        time_sec = alternative["time_sec"]

        flood_info = {
            "flooded_length_m": 0,
            "route_length_m": 0,
            "flood_percentage": 0,
            "flood_status": "Not assessed"
        }

        if scenario == "Flood + Traffic":

            flood_info = calculate_route_flood_exposure(
                graph,
                route,
                flood_geometry
            )

        route_results.append({
            "route_number": i,
            "route": route,
            "time_sec": time_sec,
            "time_min": time_sec / 60,
            "flooded_length_m": flood_info["flooded_length_m"],
            "route_length_m": flood_info["route_length_m"],
            "flood_percentage": flood_info["flood_percentage"],
            "flood_status": flood_info["flood_status"],
        })

    routes_df = pd.DataFrame(route_results)

    if routes_df.empty:
        return None

    best_route = routes_df.iloc[0]

    recommendation = route_recommendation(
        routes_df,
        scenario
    )

    return {
        "incident_id": incident_id,
        "incident_node": incident_node,
        "scenario": scenario,
        "hospital": best_hospital_name,
        "hospital_node": best_hospital_node,
        "hospital_time_min": (
            best_hospital["travel_time_sec"] / 60
        ),
        "hospital_rankings": hospital_df,
        "routes": routes_df,
        "best_route": best_route,
        "recommendation": recommendation,
    }


# ------------------------------------------------------------
# MAP HELPERS
# ------------------------------------------------------------

def flood_status_from_pct(pct):
    if pct <= 0:
        return "Not affected"
    elif pct <= 20:
        return "Minor exposure"
    elif pct <= 50:
        return "Moderate disruption"
    return "Major disruption"


def build_base_map(center, bounds=None):

    m = folium.Map(
        location=center,
        zoom_start=14,
        control_scale=True,
        tiles=None
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri",
        name="Light",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri",
        name="Dark",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="Streets",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri",
        name="Satellite",
        control=True,
    ).add_to(m)

    plugins.Fullscreen(
        position="topleft",
        title="Fullscreen",
        title_cancel="Exit fullscreen"
    ).add_to(m)

    plugins.MiniMap(
        toggle_display=True,
        position="bottomright",
        zoom_level_offset=-5
    ).add_to(m)

    if bounds:
        m.fit_bounds(
            bounds,
            padding=(30, 30)
        )

    return m


def beautified_marker(
    lat,
    lon,
    icon,
    color,
    popup=None,
    tooltip=None
):
    return folium.Marker(
        [lat, lon],
        popup=popup,
        tooltip=tooltip,
        icon=BeautifyIcon(
            icon=icon,
            icon_shape="marker",
            border_color=color,
            text_color=WHITE,
            background_color=color
        ),
    )


def compute_scenario_bounds(
    incident_latlon,
    hospital_latlon,
    route_gdfs
):
    """Bounds covering incident, facility and candidate routes."""

    lats = [
        incident_latlon[0],
        hospital_latlon[0]
    ]

    lons = [
        incident_latlon[1],
        hospital_latlon[1]
    ]

    for gdf in route_gdfs:

        if gdf.empty:
            continue

        b = gdf.total_bounds

        lons += [b[0], b[2]]
        lats += [b[1], b[3]]

    return [
        [min(lats), min(lons)],
        [max(lats), max(lons)]
    ]


# ------------------------------------------------------------
# FLOOD RASTER OVERLAYS
# ------------------------------------------------------------

flood_bounds_wgs84 = transform_bounds(
    flood_raster.crs,
    "EPSG:4326",
    *flood_raster.bounds
)

sus_rgba = None
susceptibility_bounds_wgs84 = None

if susceptibility_raster is not None:

    susceptibility_bounds_wgs84 = transform_bounds(
        susceptibility_raster.crs,
        "EPSG:4326",
        *susceptibility_raster.bounds
    )

    sus_data = susceptibility_raster.read(1)
    sus_masked = np.ma.masked_invalid(sus_data)
    sus_norm = np.clip(
        sus_masked,
        0,
        1
    )

    sus_rgba = plt.cm.YlOrRd(
        sus_norm.filled(0)
    )

    sus_rgba[..., 3] = np.where(
        sus_masked.mask,
        0,
        0.60
    )

flood_data = flood_raster.read(1)
flood_mask = flood_data == 1

flood_rgba = np.zeros(
    (
        flood_data.shape[0],
        flood_data.shape[1],
        4
    )
)

flood_rgba[flood_mask] = [
    0.90,
    0.13,
    0.27,
    0.55
]


# ------------------------------------------------------------
# AFFECTED ROADS
# ------------------------------------------------------------

affected_roads = edges_projected[
    edges_projected["flood_percentage"] > 0
].copy()


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.markdown(
    f"""
    <div style="
        padding:18px 22px;
        border-radius:10px;
        background:{NAVY};
        margin-bottom:18px;">
        <div style="
            color:{TEAL};
            font-size:11px;
            font-weight:700;
            letter-spacing:1.5px;
            margin-bottom:4px;">
            URIP · URBAN RESILIENCE INTELLIGENCE PLATFORM
        </div>
        <h1 style="
            margin:0;
            color:{WHITE};
            font-size:28px;">
            Emergency Response Dashboard
        </h1>
        <p style="
            margin:6px 0 0 0;
            color:#C8D4E0;
            font-size:14px;">
            Select an incident and a scenario to evaluate the
            best emergency response route.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# CONTROLS & SESSION STATE (STRICT BUTTON-ONLY TRIGGER)
# ------------------------------------------------------------

# Build Incident list with a placeholder
incident_options = ["Select Incident..."] + sorted(
    incidents_wgs84["incident_id"]
    .astype(int)
    .astype(str)
    .tolist()
)

# Build Scenario list with a placeholder
scenario_options = [
    "Select Scenario...",
    "Off-Peak",
    "Peak Traffic",
    "Flood + Traffic"
]

col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    selected_inc_str = st.selectbox(
        "Incident",
        options=incident_options,
        index=0,
        key="selected_incident"
    )

with col2:
    selected_scenario = st.selectbox(
        "Scenario",
        options=scenario_options,
        index=0,
        key="selected_scenario"
    )

with col3:
    st.write("") # Alignment spacer
    st.write("")
    analyse = st.button(
        "Analyse Incident",
        type="primary",
        use_container_width=True
    )

# 1. Update active analysis and parameters ONLY when the button is explicitly pressed
if analyse:
    if selected_inc_str == "Select Incident..." or selected_scenario == "Select Scenario...":
        st.warning("⚠️ Please select both an Incident ID and a Scenario before analyzing.")
    else:
        # Store the active run parameters in session state
        st.session_state["active_incident_id"] = int(selected_inc_str)
        st.session_state["active_scenario"] = selected_scenario
        
        # Execute analysis and store results
        st.session_state["analysis"] = analyse_scenario(
            st.session_state["active_incident_id"],
            st.session_state["active_scenario"],
            k_routes=3
        )

# 2. Retrieve the parameters that were LAST analyzed (not the ones currently in the dropdown)
incident_id = st.session_state.get("active_incident_id", None)
scenario = st.session_state.get("active_scenario", None)
analysis = st.session_state.get("analysis", None)

# 3. Halt rendering if no analysis has been executed yet
if analysis is None or incident_id is None or scenario is None:
    st.info("👈 Select an incident and scenario above, then click **Analyse Incident** to view emergency routes.")
    st.stop()

# Safe row extraction for downstream map/metric rendering (line 927)
matched_incidents = incidents_wgs84[
    incidents_wgs84["incident_id"].astype(str) == str(incident_id)
]

if matched_incidents.empty:
    st.error(f"Incident ID {incident_id} was not found in the dataset.")
    st.stop()

incident_row = matched_incidents.iloc[0]


# ------------------------------------------------------------
# EXTRACT RESULTS
# ------------------------------------------------------------

incident_row = incidents_wgs84[
    incidents_wgs84["incident_id"] == incident_id
].iloc[0]

incident_lat = incident_row.geometry.y
incident_lon = incident_row.geometry.x

hospital_name = analysis["hospital"]

hospital_row = hospitals_cbd[
    hospitals_cbd["name"] == hospital_name
].iloc[0]

hospital_lat = hospital_row.geometry.y
hospital_lon = hospital_row.geometry.x

routes_df = analysis["routes"].copy()

recommendation = analysis["recommendation"]

best_time = analysis["hospital_time_min"]


# ------------------------------------------------------------
# SUMMARY CARDS
# ------------------------------------------------------------

rec_color = (
    "#2A9D8F"
    if recommendation.startswith("USE")
    else "#F4A261"
    if recommendation.startswith("REROUTE")
    else "#E63946"
    if recommendation.startswith("CAUTION")
    else "#9D0208"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "🏥 Best Facility",
        hospital_name
    )

with c2:
    st.metric(
        "⏱ Fastest Response",
        f"{best_time:.2f} min"
    )

with c3:
    st.markdown(
        f"""
        <div style="
            border-left:4px solid {rec_color};
            background:{CARD_BG};
            padding:12px 15px;
            border-radius:8px;">
            <div style="
                color:{MUTED};
                font-size:11px;
                font-weight:700;
                text-transform:uppercase;">
                📡 Operational Action
            </div>
            <div style="
                color:{TEXT_DARK};
                font-size:17px;
                font-weight:700;
                margin-top:5px;">
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# BUILD MAP
# ------------------------------------------------------------

graph_for_scenario = (
    G
    if scenario == "Off-Peak"
    else G_traffic
    if scenario == "Peak Traffic"
    else G_combined
)

route_gdfs = {}

for _, route_row in routes_df.iterrows():

    route_gdfs[
        int(route_row["route_number"])
    ] = route_to_gdf_safe(
        graph_for_scenario,
        route_row["route"]
    )

scenario_bounds = compute_scenario_bounds(
    (incident_lat, incident_lon),
    (hospital_lat, hospital_lon),
    list(route_gdfs.values())
)

m = build_base_map(
    center=[
        incident_lat,
        incident_lon
    ],
    bounds=scenario_bounds
)


# ------------------------------------------------------------
# FLOOD LAYERS
# ------------------------------------------------------------

if scenario == "Flood + Traffic":

    if sus_rgba is not None:

        folium.raster_layers.ImageOverlay(
            image=sus_rgba,
            bounds=[
                [
                    flood_bounds_wgs84[1],
                    flood_bounds_wgs84[0]
                ],
                [
                    flood_bounds_wgs84[3],
                    flood_bounds_wgs84[2]
                ]
            ],
            opacity=0.60,
            interactive=True,
            cross_origin=False,
            zindex=1,
            name="Flood Susceptibility",
        ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=flood_rgba,
        bounds=[
            [
                flood_bounds_wgs84[1],
                flood_bounds_wgs84[0]
            ],
            [
                flood_bounds_wgs84[3],
                flood_bounds_wgs84[2]
            ]
        ],
        opacity=0.55,
        interactive=True,
        cross_origin=False,
        zindex=2,
        name="Flood Scenario",
    ).add_to(m)

    affected_layer = folium.FeatureGroup(
        name="Flood-Affected Roads"
    )

    for _, row in affected_roads.iterrows():

        if row.geometry is None:
            continue

        status = flood_status_from_pct(
            row["flood_percentage"]
        )

        line_color = STATUS_COLORS[status]

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda feature, c=line_color: {
                "color": c,
                "weight": 2,
                "opacity": 0.7
            },
            tooltip=(
                f"Flood exposure: "
                f"{row['flood_percentage']:.1f}% "
                f"({status})"
            ),
        ).add_to(affected_layer)

    affected_layer.add_to(m)


# ------------------------------------------------------------
# INCIDENT + HOSPITAL MARKERS
# ------------------------------------------------------------

beautified_marker(
    incident_lat,
    incident_lon,
    icon="exclamation-triangle",
    color=CORAL,
    popup=(
        f"<b>Incident {incident_id}</b>"
        "<br>Simulated Emergency"
    ),
    tooltip=f"Incident {incident_id}",
).add_to(m)

beautified_marker(
    hospital_lat,
    hospital_lon,
    icon="plus",
    color=NAVY,
    popup=(
        f"<b>{hospital_name}</b>"
        f"<br>Best facility for {scenario}"
        f"<br>Response time: {best_time:.2f} min"
    ),
    tooltip=f"Best facility: {hospital_name}",
).add_to(m)


# ------------------------------------------------------------
# ROUTE LAYERS
# ------------------------------------------------------------

rank_colors = [
    TEAL,
    CORAL,
    NAVY_2
]

for _, route_row in routes_df.iterrows():

    route_number = int(
        route_row["route_number"]
    )

    route_time = route_row["time_min"]

    flood_pct = route_row["flood_percentage"]

    flood_status = route_row["flood_status"]

    if scenario == "Flood + Traffic":

        route_color = STATUS_COLORS.get(
            flood_status,
            MUTED
        )

    else:

        route_color = rank_colors[
            min(
                route_number - 1,
                len(rank_colors) - 1
            )
        ]

    route_gdf = route_gdfs[
        route_number
    ]

    route_layer = folium.FeatureGroup(
        name=(
            f"Route {route_number} — "
            f"{route_time:.2f} min"
        )
    )

    for _, edge in route_gdf.iterrows():

        folium.GeoJson(
            edge.geometry.__geo_interface__,
            style_function=lambda feature, c=route_color: {
                "color": c,
                "weight": 6,
                "opacity": 0.9
            },
            tooltip=(
                f"Route {route_number}"
                f"<br>Travel time: "
                f"{route_time:.2f} min"
                f"<br>Flood exposure: "
                f"{flood_pct:.1f}%"
                f"<br>Flood status: "
                f"{flood_status}"
            ),
        ).add_to(route_layer)

    route_layer.add_to(m)


# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

if scenario == "Flood + Traffic":

    legend_rows = ""

    for status, color in STATUS_COLORS.items():

        if status in (
            "Unknown",
            "Not assessed"
        ):
            continue

        legend_rows += (
            f'<div style="margin:3px 0;">'
            f'<span style="display:inline-block;'
            f'width:14px;height:4px;'
            f'background:{color};'
            f'margin-right:8px;'
            f'border-radius:2px;"></span>'
            f'{status}'
            f'</div>'
        )

    legend_html = f"""
    <div style="
        position:fixed;
        bottom:30px;
        left:30px;
        z-index:9999;
        background:{WHITE};
        padding:12px 14px;
        border-radius:8px;
        box-shadow:0 2px 8px rgba(11,37,69,0.25);
        font-size:12px;
        color:{TEXT_DARK};
        font-family:sans-serif;">

        <div style="
            font-weight:700;
            margin-bottom:6px;">
            Route / Road Flood Exposure
        </div>

        {legend_rows}

        <hr style="
            margin:8px 0;
            border:none;
            border-top:1px solid {CARD_BG};">

        <div>
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:{STATUS_COLORS['Moderate disruption']};
                opacity:0.6;
                margin-right:8px;
                border-radius:2px;">
            </span>
            Flood scenario extent
        </div>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )

folium.LayerControl(
    collapsed=False
).add_to(m)


# ------------------------------------------------------------
# MAP DISPLAY
# ------------------------------------------------------------

st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[]
)


# ------------------------------------------------------------
# ROUTE COMPARISON
# ------------------------------------------------------------

st.subheader("Route Comparison")

display_table = routes_df[
    [
        "route_number",
        "time_min",
        "route_length_m",
        "flooded_length_m",
        "flood_percentage",
        "flood_status"
    ]
].copy()

display_table = display_table.rename(
    columns={
        "route_number": "Route",
        "time_min": "Travel Time (min)",
        "route_length_m": "Route Length (m)",
        "flooded_length_m": "Flooded Length (m)",
        "flood_percentage": "Flood Exposure (%)",
        "flood_status": "Flood Status"
    }
)

if scenario != "Flood + Traffic":

    display_table[
        "Flood Exposure (%)"
    ] = "N/A"

    display_table[
        "Flood Status"
    ] = "Not assessed"

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# OPERATIONAL RECOMMENDATION
# ------------------------------------------------------------
with st.container(border=True):
    st.subheader("📡 Operational Recommendation")
    st.success(f"**{recommendation}**")
    st.caption("The system evaluates travel time together with network conditions and, under the flood scenario, route-level flood exposure.")


# ------------------------------------------------------------
# FACILITY RANKING
# ------------------------------------------------------------

with st.expander("Emergency Facility Ranking"):

    ranking = analysis[
        "hospital_rankings"
    ].copy()

    ranking["travel_time_min"] = (
        ranking["travel_time_sec"] / 60
    )

    ranking = ranking[
        [
            "hospital",
            "travel_time_min"
        ]
    ].rename(
        columns={
            "hospital": "Facility",
            "travel_time_min": "Response Time (min)"
        }
    )

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )
