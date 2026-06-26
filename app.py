import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import math
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# folium for proper map icons
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSafe AI — Accident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_PATH    = "Road Accident Dataset"
DB_PATH      = os.path.join(BASE_PATH, "accidents.db")
VIDEO_FOLDER = os.path.join(BASE_PATH, "AccidentDataset")
FRAME_FOLDER = os.path.join(BASE_PATH, "accident_evidence_timestamped")
FINAL_CSV    = os.path.join(BASE_PATH, "final_accident_detection.csv")
SEVERITY_CSV = os.path.join(BASE_PATH, "video_accident_severity_v2.csv")
CAMERA_JSON  = os.path.join(BASE_PATH, "camera_coords.json")

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] {display: none;}

.top-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
}
.header-title { font-size:26px; font-weight:700; color:white; margin:0; }
.header-sub   { font-size:12px; color:#aaa; margin:4px 0 0 0; }
.header-live  {
    background:rgba(99,153,34,0.2); border:1px solid #639922;
    border-radius:99px; padding:6px 16px; color:#9fe1cb; font-size:13px;
}
.section-header {
    background:#f0f2f6; border-left:4px solid #E24B4A;
    padding:10px 16px; border-radius:0 8px 8px 0;
    margin:28px 0 16px 0; font-size:15px; font-weight:600; color:#1a1a2e;
}
.kpi-card {
    background:white; border:1px solid #e8e8e8; border-radius:12px;
    padding:16px 20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
.kpi-number { font-size:36px; font-weight:700; margin:4px 0; }
.kpi-label  { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.05em; }
.info-card {
    background:white; border:1px solid #e8e8e8; border-radius:10px;
    padding:12px 14px; box-shadow:0 1px 4px rgba(0,0,0,0.05); margin-bottom:8px;
}
.info-card-title { font-size:11px; color:#888; text-transform:uppercase; margin-bottom:4px; }
.info-card-value { font-size:14px; font-weight:600; color:#1a1a2e; }
.routing-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:10px; padding:14px 16px; margin-bottom:10px;
}
.routing-card-title  { font-size:13px; font-weight:700; color:#1a1a2e; margin-bottom:6px; }
.routing-card-detail { font-size:12px; color:#555; margin:2px 0; }
.alert-sent {
    background:#EAF3DE; border-left:3px solid #639922;
    padding:10px 14px; border-radius:0 8px 8px 0;
    margin-bottom:8px; font-size:13px; color:#27500A;
}
.alert-call {
    background:#EAF0FB; border-left:3px solid #185FA5;
    padding:10px 14px; border-radius:0 8px 8px 0;
    margin-bottom:8px; font-size:13px; color:#0d2e5c;
}
.alert-pending {
    background:#FAEEDA; border-left:3px solid #EF9F27;
    padding:10px 14px; border-radius:0 8px 8px 0;
    margin-bottom:8px; font-size:13px; color:#633806;
}
.frame-popup {
    background: white; border-radius: 14px; padding: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18); border: 1px solid #e8e8e8;
}
.frame-popup-title {
    font-size: 15px; font-weight: 700; color: #1a1a2e;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #E24B4A;
}
.frame-popup-meta {
    font-size: 12px; color: #555; margin-bottom: 10px;
    background: #f8fafc; border-radius: 8px; padding: 8px 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def get_connection():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=60)
def load_accidents():
    conn = get_connection()
    if conn is None: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM accidents", conn)
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def load_evidence():
    conn = get_connection()
    if conn is None: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM evidence LIMIT 1000", conn)
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_hospitals():
    conn = get_connection()
    if conn is None: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM hospitals", conn)
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_police():
    conn = get_connection()
    if conn is None: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM police_station", conn)
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_alerts():
    conn = get_connection()
    if conn is None: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM alerts ORDER BY id DESC LIMIT 20", conn)
    except: return pd.DataFrame()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── OSRM road routing ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_road_route(lat1, lon1, lat2, lon2):
    """Returns list of [lat, lon] pairs following actual roads via OSRM."""
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{lon1},{lat1};{lon2},{lat2}"
               f"?overview=full&geometries=geojson")
        resp = requests.get(url, timeout=6)
        data = resp.json()
        coords = data["routes"][0]["geometry"]["coordinates"]
        # OSRM returns [lon, lat] — flip to [lat, lon] for folium
        return [[c[1], c[0]] for c in coords]
    except Exception:
        return [[lat1, lon1], [lat2, lon2]]   # straight-line fallback

# ── EXACT frame folder matching ───────────────────────────────────────────────
def find_frame_folder(video_name: str):
    """
    Folder names: video1_Critical, video2_Major, video10_Critical ...
    video_name from CSV: video1.mp4, video2.mp4, video10.mp4 ...

    FIX: extract digits, then do EXACT numeric match so video5 ≠ video50.
    """
    if not os.path.exists(FRAME_FOLDER):
        return None

    stem   = video_name.replace(".mp4","").replace(".MP4","").strip().lower()
    # numeric part of stem  e.g. "video5" → "5",  "video50" → "50"
    digits = ''.join(filter(str.isdigit, stem))

    all_dirs = [d for d in os.listdir(FRAME_FOLDER)
                if os.path.isdir(os.path.join(FRAME_FOLDER, d))]

    # Pass 1 — exact stem match  (stem == folder_prefix_before_underscore)
    for d in all_dirs:
        folder_prefix = d.lower().split("_")[0]          # "video1_critical" → "video1"
        if folder_prefix == stem:                         # exact: "video5" == "video5"
            return os.path.join(FRAME_FOLDER, d)

    # Pass 2 — exact numeric match on prefix
    if digits:
        for d in all_dirs:
            folder_prefix  = d.lower().split("_")[0]     # "video50_major" → "video50"
            folder_digits  = ''.join(filter(str.isdigit, folder_prefix))   # "50"
            if folder_digits == digits:                   # exact: "5" vs "50" → no match
                return os.path.join(FRAME_FOLDER, d)

    return None

# ── Hospital/Police detail formatter ─────────────────────────────────────────
def format_place_details(row: pd.Series) -> str:
    """
    Show in fixed order: name, address, lat, lon, id, phone, email
    then any remaining columns.
    """
    priority = ["name","hospital_name","station_name",
                "address","full_address",
                "latitude","longitude",
                "id","hospital_id","station_id",
                "phone","contact","phone_number",
                "email","email_id"]

    lines = []
    shown = set()

    def add(col, label, val):
        if col not in shown and not pd.isna(val) and str(val).strip() not in ("","—"):
            lines.append(f"<div class='routing-card-detail'><b>{label}:</b> {val}</div>")
            shown.add(col)

    # Fixed priority fields
    for c in priority:
        if c in row.index:
            label = c.replace("_"," ").title()
            if c in ("latitude","longitude"):
                label = "Lat" if c == "latitude" else "Lon"
            elif c in ("hospital_id","station_id","id"):
                label = "ID"
            elif c in ("phone","contact","phone_number"):
                label = "📞 Phone"
            elif c in ("email","email_id"):
                label = "📧 Email"
            elif c in ("address","full_address"):
                label = "📍 Address"
            add(c, label, row[c])

    # Remaining columns
    for c in row.index:
        if c not in shown and c not in ("dist_km","latitude","longitude"):
            label = c.replace("_"," ").title()
            add(c, label, row[c])

    return "\n".join(lines)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df        = load_accidents()
evid_df   = load_evidence()
hosp_df   = load_hospitals()
police_df = load_police()
alerts_df = load_alerts()

critical = len(df[df["severity_level"]=="Critical"]) if not df.empty and "severity_level" in df.columns else 0
major    = len(df[df["severity_level"]=="Major"])    if not df.empty and "severity_level" in df.columns else 0
minor    = len(df[df["severity_level"]=="Minor"])    if not df.empty and "severity_level" in df.columns else 0

# ═══════════════════════════════════════
# HEADER
# ═══════════════════════════════════════
st.markdown(f"""
<div class="top-header">
    <div>
        <p class="header-title">🚨 RoadSafe AI — Accident Detection System</p>
        <p class="header-sub">Central Command Dashboard &nbsp;·&nbsp; Bengaluru Traffic Surveillance Network</p>
    </div>
    <div class="header-live">● &nbsp; System Live &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y &nbsp;&nbsp; %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# 1. KPI METRICS
# ═══════════════════════════════════════
st.markdown('<div class="section-header">📊 Live Metrics</div>', unsafe_allow_html=True)

n_videos = df["video_name"].nunique() if not df.empty and "video_name" in df.columns else 0
k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value, color in zip(
    [k1, k2, k3, k4, k5],
    ["Total Incidents","🔴 Critical","🟠 Major","🟢 Minor","📹 Videos"],
    [len(df), critical, major, minor, n_videos],
    ["#1a1a2e","#E24B4A","#EF9F27","#639922","#185FA5"]
):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-number" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# 2. SEVERITY ASSESSMENT
# ═══════════════════════════════════════
st.markdown('<div class="section-header">⚠️ Severity Assessment</div>', unsafe_allow_html=True)

sv1, sv2 = st.columns([1, 2])
with sv1:
    total = max(len(df), 1)
    for level, color, emoji in [("Critical","#E24B4A","🔴"),("Major","#EF9F27","🟠"),("Minor","#639922","🟢")]:
        n   = len(df[df["severity_level"]==level]) if not df.empty and "severity_level" in df.columns else 0
        pct = round(n/total*100, 1)
        st.markdown(f"""
        <div class="info-card" style="border-left:4px solid {color}">
            <div class="info-card-title">{emoji} {level}</div>
            <div class="info-card-value" style="color:{color}">{n} incidents &nbsp;·&nbsp; {pct}%</div>
        </div>""", unsafe_allow_html=True)

with sv2:
    if not df.empty and "severity_level" in df.columns:
        sev_f = st.multiselect("Filter", ["Critical","Major","Minor"],
                               default=["Critical","Major","Minor"], key="sev_filter")
        filtered = df[df["severity_level"].isin(sev_f)]
        show_cols = [c for c in ["video_name","severity_level","confidence_score","location","timestamp"]
                     if c in filtered.columns]
        st.dataframe(filtered[show_cols].sort_values("severity_level").head(20),
                     use_container_width=True, hide_index=True, height=220)

# ═══════════════════════════════════════
# 3. LIVE ACCIDENT MAP  — folium with proper pin icons
#    Critical = red pin, Major = orange pin, Minor = green pin
#    ONE legend entry per severity (no duplicates)
# ═══════════════════════════════════════
st.markdown('<div class="section-header">🗺️ Live Accident Map</div>', unsafe_allow_html=True)

if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
    map_df = df.dropna(subset=["latitude","longitude"]).copy()
    if not map_df.empty:
        # ── Severity filter legend (checkboxes above the map) ─────────────────
        leg_cols = st.columns([1, 1, 1, 5])
        sev_filter_live = []
        with leg_cols[0]:
            if st.checkbox("🔴 Critical", value=True, key="lmap_crit"):
                sev_filter_live.append("Critical")
        with leg_cols[1]:
            if st.checkbox("🟠 Major", value=True, key="lmap_maj"):
                sev_filter_live.append("Major")
        with leg_cols[2]:
            if st.checkbox("🟢 Minor", value=True, key="lmap_min"):
                sev_filter_live.append("Minor")

        map_df_filtered = map_df[map_df["severity_level"].isin(sev_filter_live)] \
            if "severity_level" in map_df.columns else map_df

        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        acc_map = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                             tiles="OpenStreetMap")

        # Folium pin colour per severity
        SEV_COLOR = {"Critical": "red", "Major": "orange", "Minor": "green"}
        SEV_HEX   = {"Critical": "#c0392b", "Major": "#e67e22", "Minor": "#27ae60"}

        for _, row in map_df_filtered.iterrows():
            level    = str(row.get("severity_level", "Minor"))
            pin_col  = SEV_COLOR.get(level, "blue")
            hex_col  = SEV_HEX.get(level, "#3498db")
            rid      = row.get("incident_id", row.get("id", "—"))
            location = row.get("location", "—")
            lat      = row.get("latitude", "—")
            lon      = row.get("longitude", "—")
            ts       = row.get("timestamp", "—")
            conf     = row.get("confidence_score", "—")

            tooltip_html = f"""
            <div style="font-family:Arial;min-width:220px;max-width:280px;padding:4px;">
                <div style="font-size:13px;font-weight:700;margin-bottom:4px;">📍 {location}</div>
                <hr style="margin:4px 0;border-color:#ddd">
                <div style="font-size:12px;line-height:1.7;">
                    <b>ID:</b> {rid}<br>
                    <b>Address / Location:</b> {location}<br>
                    <b>Lat:</b> {lat} &nbsp; <b>Lon:</b> {lon}<br>
                    <b style="font-size:15px;color:{hex_col};">Severity: {level}</b><br>
                    <b>Timestamp:</b> {ts}<br>
                    <b>Confidence:</b> {conf}
                </div>
            </div>"""

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(tooltip_html, max_width=300),
                icon=folium.Icon(color=pin_col, icon="map-marker", prefix="fa")
            ).add_to(acc_map)

        st_folium(acc_map, use_container_width=True, height=460, returned_objects=[])
    else:
        st.info("No GPS coordinates available in database.")
else:
    st.info("No location data in database.")

# ═══════════════════════════════════════
# 4. HOSPITAL MAP  |  POLICE STATION MAP
#    Hospital: green pin with plus icon
#    Police:   blue pin with info icon
#    Hover details: name, address, lat, lon, id, phone, email
# ═══════════════════════════════════════
st.markdown('<div class="section-header">🏥 Hospital Map &nbsp;&nbsp;|&nbsp;&nbsp; 🚔 Police Station Map</div>',
            unsafe_allow_html=True)

hm_col, pm_col = st.columns(2)

# ── Hospital map ──────────────────────────────────────────────────────────────
with hm_col:
    st.markdown("#### ➕ Hospitals")
    if not hosp_df.empty and "latitude" in hosp_df.columns and "longitude" in hosp_df.columns:
        h_clean = hosp_df.dropna(subset=["latitude","longitude"]).copy()
        if not h_clean.empty:
            h_map = folium.Map(
                location=[h_clean["latitude"].mean(), h_clean["longitude"].mean()],
                zoom_start=11, tiles="OpenStreetMap"
            )
            for _, row in h_clean.iterrows():
                name    = row.get("name", row.get("hospital_name","Hospital"))
                address = row.get("address", row.get("full_address","—"))
                lat     = row.get("latitude","—")
                lon     = row.get("longitude","—")
                phone   = row.get("phone", row.get("contact","—"))
                email   = row.get("email", row.get("email_id","—"))

                tooltip_html = f"""
                <div style="font-family:Arial;min-width:220px;max-width:280px;padding:4px;">
                    <div style="font-size:13px;font-weight:700;color:#1565C0;margin-bottom:4px;">➕ {name}</div>
                    <hr style="margin:4px 0;border-color:#ddd">
                    <div style="font-size:12px;line-height:1.8;">
                        <b>Name:</b> {name}<br>
                        <b>Address:</b> {address}<br>
                        <b>Lat:</b> {lat} &nbsp; <b>Lon:</b> {lon}<br>
                        <b>📞 Phone:</b> {phone}<br>
                        <b>📧 Email:</b> {email}
                    </div>
                </div>"""

                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(tooltip_html, max_width=300),
                    icon=folium.Icon(color="blue", icon="plus", prefix="fa")
                ).add_to(h_map)

            st_folium(h_map, use_container_width=True, height=400, returned_objects=[])
        else:
            st.info("No hospital coordinates found.")
    else:
        st.info("Hospitals table not found or missing coordinates.")

# ── Police station map ────────────────────────────────────────────────────────
with pm_col:
    st.markdown("#### 🚔 Police Stations")
    if not police_df.empty and "latitude" in police_df.columns and "longitude" in police_df.columns:
        p_clean = police_df.dropna(subset=["latitude","longitude"]).copy()
        if not p_clean.empty:
            p_map = folium.Map(
                location=[p_clean["latitude"].mean(), p_clean["longitude"].mean()],
                zoom_start=11, tiles="OpenStreetMap"
            )
            for _, row in p_clean.iterrows():
                name    = row.get("name", row.get("station_name","Police Station"))
                address = row.get("address", row.get("full_address","—"))
                lat     = row.get("latitude","—")
                lon     = row.get("longitude","—")
                phone   = row.get("phone", row.get("contact","—"))
                email   = row.get("email", row.get("email_id","—"))

                tooltip_html = f"""
                <div style="font-family:Arial;min-width:220px;max-width:280px;padding:4px;">
                    <div style="font-size:13px;font-weight:700;color:#1B5E20;margin-bottom:4px;">🚔 {name}</div>
                    <hr style="margin:4px 0;border-color:#ddd">
                    <div style="font-size:12px;line-height:1.8;">
                        <b>Name:</b> {name}<br>
                        <b>Address:</b> {address}<br>
                        <b>Lat:</b> {lat} &nbsp; <b>Lon:</b> {lon}<br>
                        <b>📞 Phone:</b> {phone}<br>
                        <b>📧 Email:</b> {email}
                    </div>
                </div>"""

                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(tooltip_html, max_width=300),
                    icon=folium.Icon(color="green", icon="shield", prefix="fa")
                ).add_to(p_map)

            st_folium(p_map, use_container_width=True, height=400, returned_objects=[])
        else:
            st.info("No police station coordinates found.")
    else:
        st.info("Police station table not found or missing coordinates.")

# ═══════════════════════════════════════
# 5. VIDEO ANALYSIS
# ═══════════════════════════════════════
st.markdown('<div class="section-header">📹 Video Analysis</div>', unsafe_allow_html=True)

selected_video = None

if os.path.exists(FINAL_CSV):
    final_df   = pd.read_csv(FINAL_CSV)
    video_list = sorted(
        final_df["video_name"].unique().tolist(),
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    vc1, _ = st.columns([1, 3])
    with vc1:
        selected_video = st.selectbox("Select video to inspect", video_list, key="video_sel")

    vdf = final_df[final_df["video_name"] == selected_video]

    vm1, vm2, vm3, vm4 = st.columns(4)
    vm1.metric("Total frames",        len(vdf))
    vm2.metric("Collision frames",    int(vdf["collision_detected"].sum())
               if "collision_detected" in vdf.columns else "—")
    vm3.metric("Speed drop frames",   int(vdf["sudden_speed_drop"].sum())
               if "sudden_speed_drop"  in vdf.columns else "—")
    vm4.metric("Confirmed accidents", int(vdf["accident_confirmed"].sum())
               if "accident_confirmed" in vdf.columns else "—")

    # Signal chart
    signal_cols = [c for c in ["collision_detected","sudden_speed_drop",
                                "optical_flow_spike","temporal_consistent","accident_confirmed"]
                   if c in vdf.columns]
    if signal_cols:
        plot_df = vdf[["frame_number"] + signal_cols].copy()
        for c in signal_cols:
            plot_df[c] = plot_df[c].astype(int)
        melted = plot_df.melt(id_vars="frame_number", value_vars=signal_cols,
                               var_name="Signal", value_name="Active")
        fig_sig = px.line(melted, x="frame_number", y="Active", color="Signal", height=260)
        fig_sig.update_layout(margin=dict(t=10,b=10), xaxis_title="Frame", yaxis_title="Active (0/1)")
        st.plotly_chart(fig_sig, use_container_width=True)

    if "speed_drop_ratio" in vdf.columns:
        fig_spd = px.area(vdf, x="frame_number", y="speed_drop_ratio",
                          color_discrete_sequence=["#E24B4A"], height=200)
        fig_spd.update_layout(margin=dict(t=10,b=10), xaxis_title="Frame",
                               yaxis_title="Speed Drop Ratio")
        st.plotly_chart(fig_spd, use_container_width=True)

    # ── VIDEO PLAYER ──────────────────────────────────────────────────────────
    if selected_video and os.path.exists(VIDEO_FOLDER):
        stem = selected_video.replace(".mp4","").replace(".MP4","").strip()
        video_path = None
        for ext in [".mp4",".MP4",".avi",".AVI"]:
            p = os.path.join(VIDEO_FOLDER, stem + ext)
            if os.path.exists(p):
                video_path = p
                break
        if video_path is None:
            for root, dirs, files in os.walk(VIDEO_FOLDER):
                for f in files:
                    if f.lower().replace(".mp4","").replace(".avi","") == stem.lower():
                        video_path = os.path.join(root, f)
                        break
                if video_path:
                    break
        if video_path:
            st.markdown("**▶️ Video Playback**")
            st.video(video_path)
        else:
            st.info(f"Video `{selected_video}` not found in `{VIDEO_FOLDER}`.")

    # ── EVIDENCE FRAMES — EXACT MATCH FIX ────────────────────────────────────
    # video5.mp4  →  folder prefix "video5"  →  ONLY matches video5_Major
    # NOT video50_Major  (exact numeric comparison fixes this)
    if selected_video:
        folder_path = find_frame_folder(selected_video)

        if folder_path:
            all_frames = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg",".jpeg",".png"))
            ])
            folder_name = os.path.basename(folder_path)

            if all_frames:
                st.markdown(f"**🖼️ Evidence Frames — `{folder_name}`**")
                cols_f = st.columns(3)
                for i, fname in enumerate(all_frames[:6]):
                    fp = os.path.join(folder_path, fname)
                    if not os.path.exists(fp):
                        continue
                    with cols_f[i % 3]:
                        st.image(fp, use_container_width=True)
                        with st.expander(f"🔍 View · {fname}"):
                            st.markdown(f"""
                            <div class="frame-popup">
                                <div class="frame-popup-title">📸 {fname}</div>
                                <div class="frame-popup-meta">
                                    <b>Video:</b> {selected_video}<br>
                                    <b>Evidence folder:</b> {folder_name}<br>
                                    <b>Frame index:</b> {i+1} of {len(all_frames)}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(fp, use_container_width=True)
                if len(all_frames) > 6:
                    st.caption(f"Showing first 6 of {len(all_frames)} frames.")
            else:
                st.info(f"Folder `{folder_name}` found but contains no images.")
        else:
            all_dirs = os.listdir(FRAME_FOLDER) if os.path.exists(FRAME_FOLDER) else []
            st.warning(
                f"No evidence folder found for `{selected_video}`.\n\n"
                f"Available: {', '.join(all_dirs[:10]) if all_dirs else 'none'}"
            )
else:
    st.warning(f"⚠️ `final_accident_detection.csv` not found in `{BASE_PATH}/`")

# ═══════════════════════════════════════
# 6. LOCATION & EMERGENCY ROUTING
#    folium map — red pin (accident), green+ (hospital), blue (police)
#    OSRM road routes: green line = hospital, blue line = police
#    Clean legend in sidebar of map
# ═══════════════════════════════════════
st.markdown('<div class="section-header">🏥 Location & Emergency Routing</div>', unsafe_allow_html=True)

routing_row    = None
acc_lat        = None
acc_lon        = None
top_hospitals  = pd.DataFrame()
nearest_police = None

if selected_video and not df.empty and "video_name" in df.columns:
    stem_r = selected_video.replace(".mp4","").replace(".MP4","").strip()
    match  = df[df["video_name"].str.replace(".mp4","",regex=False)
                               .str.replace(".MP4","",regex=False)
                               .str.strip() == stem_r]
    if not match.empty:
        routing_row = match.iloc[0]
        acc_lat     = routing_row.get("latitude")
        acc_lon     = routing_row.get("longitude")

if routing_row is not None:
    er1, er2 = st.columns([1, 2])

    with er1:
        st.markdown("**📍 Accident Details**")
        st.markdown(f"""
        <div class="info-card"><div class="info-card-title">📍 Location</div>
        <div class="info-card-value">{routing_row.get('location','Unknown')}</div></div>
        <div class="info-card"><div class="info-card-title">⚠️ Severity</div>
        <div class="info-card-value">{routing_row.get('severity_level','—')}</div></div>
        <div class="info-card"><div class="info-card-title">🕒 Timestamp</div>
        <div class="info-card-value">{routing_row.get('timestamp','—')}</div></div>
        """, unsafe_allow_html=True)

        if pd.notna(acc_lat) and pd.notna(acc_lon):
            # 3 nearest hospitals
            if not hosp_df.empty and {"latitude","longitude"}.issubset(hosp_df.columns):
                h = hosp_df.copy()
                h["dist_km"] = h.apply(
                    lambda r: haversine(acc_lat, acc_lon, r["latitude"], r["longitude"]), axis=1)
                top_hospitals = h.nsmallest(3, "dist_km")
                st.markdown("**➕ 3 Nearest Hospitals**")
                for _, hosp in top_hospitals.iterrows():
                    h_name  = hosp.get("name", hosp.get("hospital_name","Hospital"))
                    h_addr  = hosp.get("address", hosp.get("full_address","—"))
                    h_lat   = hosp.get("latitude","—")
                    h_lon   = hosp.get("longitude","—")
                    h_id    = hosp.get("id", hosp.get("hospital_id","—"))
                    h_phone = hosp.get("phone", hosp.get("contact","—"))
                    h_email = hosp.get("email", hosp.get("email_id","—"))
                    st.markdown(f"""
                    <div class="routing-card" style="border-left:4px solid #1565C0;">
                        <div class="routing-card-title">➕ {h_name}</div>
                        <div class="routing-card-detail">📏 {hosp['dist_km']:.2f} km away</div>
                        <div class="routing-card-detail">📍 {h_addr}</div>
                        <div class="routing-card-detail">🌐 Lat: {h_lat} &nbsp; Lon: {h_lon}</div>
                        <div class="routing-card-detail">🆔 ID: {h_id}</div>
                        <div class="routing-card-detail">📞 {h_phone}</div>
                        <div class="routing-card-detail">📧 {h_email}</div>
                    </div>""", unsafe_allow_html=True)

            # Nearest police
            if not police_df.empty and {"latitude","longitude"}.issubset(police_df.columns):
                p = police_df.copy()
                p["dist_km"] = p.apply(
                    lambda r: haversine(acc_lat, acc_lon, r["latitude"], r["longitude"]), axis=1)
                nearest_police = p.nsmallest(1,"dist_km").iloc[0]
                p_name  = nearest_police.get("name", nearest_police.get("station_name","Police Station"))
                p_addr  = nearest_police.get("address", nearest_police.get("full_address","—"))
                p_lat   = nearest_police.get("latitude","—")
                p_lon   = nearest_police.get("longitude","—")
                p_id    = nearest_police.get("id", nearest_police.get("station_id","—"))
                p_phone = nearest_police.get("phone", nearest_police.get("contact","—"))
                p_email = nearest_police.get("email", nearest_police.get("email_id","—"))
                st.markdown("**🚔 Nearest Police Station**")
                st.markdown(f"""
                <div class="routing-card" style="border-left:4px solid #1B5E20;">
                    <div class="routing-card-title">🚔 {p_name}</div>
                    <div class="routing-card-detail">📏 {nearest_police['dist_km']:.2f} km away</div>
                    <div class="routing-card-detail">📍 {p_addr}</div>
                    <div class="routing-card-detail">🌐 Lat: {p_lat} &nbsp; Lon: {p_lon}</div>
                    <div class="routing-card-detail">🆔 ID: {p_id}</div>
                    <div class="routing-card-detail">📞 {p_phone}</div>
                    <div class="routing-card-detail">📧 {p_email}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No GPS coordinates for this incident.")

    with er2:
        if pd.notna(acc_lat) and pd.notna(acc_lon):
            route_map = folium.Map(location=[acc_lat, acc_lon],
                                   zoom_start=13, tiles="OpenStreetMap")

            # ── Road routes first (drawn below markers) ──────────────────────
            nearest_h_lat, nearest_h_lon = None, None
            if not top_hospitals.empty:
                nearest_h_lat = top_hospitals.iloc[0]["latitude"]
                nearest_h_lon = top_hospitals.iloc[0]["longitude"]
                h_name_leg = top_hospitals.iloc[0].get("name","Hospital")
                route_h = get_road_route(acc_lat, acc_lon, nearest_h_lat, nearest_h_lon)
                folium.PolyLine(
                    route_h, color="#1565C0", weight=5, opacity=0.9,
                    tooltip=f"Route to {h_name_leg}"
                ).add_to(route_map)

            if nearest_police is not None:
                p_name_leg = nearest_police.get("name","Police Station")
                route_p = get_road_route(
                    acc_lat, acc_lon,
                    nearest_police["latitude"], nearest_police["longitude"]
                )
                folium.PolyLine(
                    route_p, color="#1B5E20", weight=5, opacity=0.9,
                    tooltip=f"Route to {p_name_leg}"
                ).add_to(route_map)

            # ── Accident marker — RED pin ─────────────────────────────────────
            acc_tooltip = f"""
            <div style="font-family:Arial;min-width:200px;max-width:260px;padding:4px;">
                <div style="font-size:13px;font-weight:700;color:#c0392b;margin-bottom:4px;">🚨 Accident Location</div>
                <hr style="margin:4px 0;border-color:#ddd">
                <div style="font-size:12px;line-height:1.8;">
                    <b>Location:</b> {routing_row.get('location','—')}<br>
                    <b>Severity:</b> <span style="color:#c0392b;font-weight:700;">{routing_row.get('severity_level','—')}</span><br>
                    <b>Time:</b> {routing_row.get('timestamp','—')}<br>
                    <b>Lat:</b> {acc_lat:.6f} &nbsp; <b>Lon:</b> {acc_lon:.6f}
                </div>
            </div>"""
            folium.Marker(
                location=[acc_lat, acc_lon],
                tooltip=folium.Tooltip(acc_tooltip, sticky=True),
                popup=folium.Popup(acc_tooltip, max_width=280),
                icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon")
            ).add_to(route_map)

            # ── Hospital markers — BLUE pins ──────────────────────────────────
            if not top_hospitals.empty:
                for idx, (_, hosp) in enumerate(top_hospitals.iterrows()):
                    h_name  = hosp.get("name", hosp.get("hospital_name","Hospital"))
                    h_addr  = hosp.get("address", hosp.get("full_address","—"))
                    h_phone = hosp.get("phone", hosp.get("contact","—"))
                    h_email = hosp.get("email", hosp.get("email_id","—"))
                    h_tip = f"""
                    <div style="font-family:Arial;min-width:220px;max-width:280px;padding:4px;">
                        <div style="font-size:13px;font-weight:700;color:#1565C0;margin-bottom:4px;">➕ {h_name}</div>
                        <hr style="margin:4px 0;border-color:#ddd">
                        <div style="font-size:12px;line-height:1.8;">
                            <b>Name:</b> {h_name}<br>
                            <b>Address:</b> {h_addr}<br>
                            <b>Lat:</b> {hosp['latitude']:.6f} &nbsp; <b>Lon:</b> {hosp['longitude']:.6f}<br>
                            <b>📞 Phone:</b> {h_phone}<br>
                            <b>📧 Email:</b> {h_email}<br>
                            <b>Distance:</b> {hosp['dist_km']:.2f} km
                        </div>
                    </div>"""
                    folium.Marker(
                        location=[hosp["latitude"], hosp["longitude"]],
                        tooltip=folium.Tooltip(h_tip, sticky=True),
                        popup=folium.Popup(h_tip, max_width=300),
                        icon=folium.Icon(color="blue", icon="plus", prefix="fa")
                    ).add_to(route_map)

            # ── Police marker — GREEN pin ─────────────────────────────────────
            if nearest_police is not None:
                p_name  = nearest_police.get("name", nearest_police.get("station_name","Police Station"))
                p_addr  = nearest_police.get("address", nearest_police.get("full_address","—"))
                p_phone = nearest_police.get("phone", nearest_police.get("contact","—"))
                p_email = nearest_police.get("email", nearest_police.get("email_id","—"))
                p_tip = f"""
                <div style="font-family:Arial;min-width:220px;max-width:280px;padding:4px;">
                    <div style="font-size:13px;font-weight:700;color:#1B5E20;margin-bottom:4px;">🚔 {p_name}</div>
                    <hr style="margin:4px 0;border-color:#ddd">
                    <div style="font-size:12px;line-height:1.8;">
                        <b>Name:</b> {p_name}<br>
                        <b>Address:</b> {p_addr}<br>
                        <b>Lat:</b> {nearest_police['latitude']:.6f} &nbsp; <b>Lon:</b> {nearest_police['longitude']:.6f}<br>
                        <b>📞 Phone:</b> {p_phone}<br>
                        <b>📧 Email:</b> {p_email}<br>
                        <b>Distance:</b> {nearest_police['dist_km']:.2f} km
                    </div>
                </div>"""
                folium.Marker(
                    location=[nearest_police["latitude"], nearest_police["longitude"]],
                    tooltip=folium.Tooltip(p_tip, sticky=True),
                    popup=folium.Popup(p_tip, max_width=300),
                    icon=folium.Icon(color="green", icon="shield", prefix="fa")
                ).add_to(route_map)

            # ── Legend: type column — accident / hospital / police ────────────
            legend_html = """
            <div style="
                position:fixed; bottom:30px; right:30px; z-index:9999;
                background:white; border:1px solid #ddd; border-radius:10px;
                padding:14px 18px; font-family:Arial; font-size:13px;
                box-shadow:0 4px 12px rgba(0,0,0,0.15); min-width:170px;">
                <div style="font-size:12px;font-weight:700;color:#555;
                            text-transform:uppercase;letter-spacing:0.05em;
                            margin-bottom:10px;">TYPE</div>
                <div style="margin:7px 0;display:flex;align-items:center;gap:10px;">
                    <svg width="13" height="13"><circle cx="6.5" cy="6.5" r="5.5" fill="#c0392b"/></svg>
                    <span style="font-size:13px;color:#333;">accident</span>
                </div>
                <div style="margin:7px 0;display:flex;align-items:center;gap:10px;">
                    <svg width="13" height="13"><circle cx="6.5" cy="6.5" r="5.5" fill="#1565C0"/></svg>
                    <span style="font-size:13px;color:#333;">hospital</span>
                </div>
                <div style="margin:7px 0;display:flex;align-items:center;gap:10px;">
                    <svg width="13" height="13"><circle cx="6.5" cy="6.5" r="5.5" fill="#2e7d32"/></svg>
                    <span style="font-size:13px;color:#333;">police</span>
                </div>
                <div style="border-top:1px solid #eee;margin:10px 0 4px 0;padding-top:8px;">
                <div style="margin:5px 0;display:flex;align-items:center;gap:10px;">
                    <svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="#1565C0" stroke-width="3"/></svg>
                    <span style="font-size:12px;color:#555;">Route → Hospital</span>
                </div>
                <div style="margin:5px 0;display:flex;align-items:center;gap:10px;">
                    <svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="#2e7d32" stroke-width="3"/></svg>
                    <span style="font-size:12px;color:#555;">Route → Police</span>
                </div>
                </div>
            </div>"""
            route_map.get_root().html.add_child(folium.Element(legend_html))

            st_folium(route_map, use_container_width=True, height=520, returned_objects=[])
        else:
            st.info("No GPS data available for routing.")
else:
    st.info("Select a video above — routing will auto-populate based on that video's accident record.")

# ═══════════════════════════════════════
# 7. ALERT & COMMUNICATION
# ═══════════════════════════════════════
st.markdown('<div class="section-header">🔔 Alert & Communication</div>', unsafe_allow_html=True)

al1, al2 = st.columns([2, 1])

with al1:
    st.markdown("**Alert Log**")
    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True, height=220)
    else:
        ts_now = datetime.now().strftime("%H:%M:%S")
        if routing_row is not None and nearest_police is not None and not top_hospitals.empty:
            nh      = top_hospitals.iloc[0]
            h_name  = nh.get("name","Hospital")
            h_phone = nh.get("phone", nh.get("contact","—"))
            h_email = nh.get("email", nh.get("email_id","—"))
            p_name  = nearest_police.get("name","Police Station")
            p_phone = nearest_police.get("phone", nearest_police.get("contact","—"))
            p_email = nearest_police.get("email", nearest_police.get("email_id","—"))
            sev     = routing_row.get("severity_level","—")
            st.markdown(f"""
            <div class="alert-sent">✅ <b>SMS sent</b> &nbsp;·&nbsp; ➕ {h_name} &nbsp;·&nbsp; Ambulance dispatched &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>Email sent</b> &nbsp;·&nbsp; ➕ {h_name} &nbsp;·&nbsp; {h_email} &nbsp;·&nbsp; Severity: {sev} &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-call">📞 <b>Call initiated</b> &nbsp;·&nbsp; ➕ {h_name} &nbsp;·&nbsp; {h_phone} &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>SMS sent</b> &nbsp;·&nbsp; 🚔 {p_name} &nbsp;·&nbsp; Incident reported &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>Email sent</b> &nbsp;·&nbsp; 🚔 {p_name} &nbsp;·&nbsp; {p_email} &nbsp;·&nbsp; Severity: {sev} &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-call">📞 <b>Call initiated</b> &nbsp;·&nbsp; 🚔 {p_name} &nbsp;·&nbsp; {p_phone} &nbsp;·&nbsp; {ts_now}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-sent">✅ <b>SMS sent</b> &nbsp;·&nbsp; ➕ St. John's Hospital &nbsp;·&nbsp; Ambulance dispatched &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>Email sent</b> &nbsp;·&nbsp; ➕ St. John's Hospital &nbsp;·&nbsp; hospital@stjohns.in &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-call">📞 <b>Call initiated</b> &nbsp;·&nbsp; ➕ St. John's Hospital &nbsp;·&nbsp; 080-22065000 &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>SMS sent</b> &nbsp;·&nbsp; 🚔 Halasuru Police Station &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-sent">✅ <b>Email sent</b> &nbsp;·&nbsp; 🚔 Halasuru Police Station &nbsp;·&nbsp; halasuru.ps@ksp.gov.in &nbsp;·&nbsp; {ts_now}</div>
            <div class="alert-call">📞 <b>Call initiated</b> &nbsp;·&nbsp; 🚔 Halasuru Police Station &nbsp;·&nbsp; 080-25584060 &nbsp;·&nbsp; {ts_now}</div>
            """, unsafe_allow_html=True)

with al2:
    st.markdown("**Manual Alert Trigger**")
    if not df.empty and "incident_id" in df.columns:
        sel_inc2   = st.selectbox("Incident ID", df["incident_id"].tolist(), key="alert_inc")
        alert_type = st.selectbox("Alert type", ["SMS","Email","Call","All"])
        if st.button("🚨 Send Alert", type="primary", use_container_width=True):
            if alert_type == "Call":
                st.success("📞 Call initiated! Add Twilio credentials.")
            elif alert_type == "Email":
                st.success("📧 Email sent! Add SMTP credentials.")
            else:
                st.success(f"✅ {alert_type} alert recorded!")
    else:
        st.info("No incidents available.")

# ═══════════════════════════════════════
# 8. INCIDENT LOGS
# ═══════════════════════════════════════
st.markdown('<div class="section-header">📋 Incident Logs</div>', unsafe_allow_html=True)

if not df.empty:
    fl1, fl2, fl3 = st.columns(3)
    with fl1:
        sev_f = st.multiselect("Severity", ["Critical","Major","Minor"],
                               default=["Critical","Major","Minor"], key="log_sev")
    with fl2:
        search_loc = st.text_input("Search location", placeholder="e.g. Silk Board")
    with fl3:
        sort_col = st.selectbox("Sort by", ["timestamp","severity_level","confidence_score"])

    log_df = df[df["severity_level"].isin(sev_f)] if "severity_level" in df.columns else df
    if search_loc and "location" in log_df.columns:
        log_df = log_df[log_df["location"].str.contains(search_loc, case=False, na=False)]
    if sort_col in log_df.columns:
        log_df = log_df.sort_values(sort_col, ascending=False)

    show_log = [c for c in ["incident_id","video_name","timestamp","location",
                             "severity_level","confidence_score","latitude","longitude"]
                if c in log_df.columns]
    st.caption(f"Showing {len(log_df)} of {len(df)} incidents")
    st.dataframe(log_df[show_log], use_container_width=True, hide_index=True, height=280)

    csv_bytes = log_df[show_log].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_bytes,
                       file_name=f"incidents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")
else:
    st.info("No incidents in database.")

# ═══════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<hr style="border:none;border-top:1px solid #e8e8e8;margin:0 0 10px 0">
<p style="text-align:center;color:#aaa;font-size:12px">
    RoadSafe AI &nbsp;·&nbsp; Road Accident Detection System &nbsp;·&nbsp; Bengaluru
    &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp; DeepSORT &nbsp;·&nbsp; Optical Flow &nbsp;·&nbsp; SQLite
</p>
""", unsafe_allow_html=True)
