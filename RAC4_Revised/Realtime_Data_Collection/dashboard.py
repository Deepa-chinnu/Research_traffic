"""
=============================================================================
REAL-TIME TRAFFIC DATA COLLECTION DASHBOARD
=============================================================================
Interactive Streamlit dashboard to visualize, collect, and STORE real-time
traffic data from TomTom & OpenWeatherMap APIs for 16 Bangalore roads.

Every fetch is SAVED to raw_traffic_data.csv permanently.
Auto-collection runs every 15 minutes in-browser.

Usage:
  streamlit run dashboard.py

Author: PhD Research - Traffic Flow Prediction
=============================================================================
"""

import os
import sys
import csv
import time
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    TOMTOM_API_KEY, OPENWEATHER_API_KEY,
    ROAD_LOCATIONS, RAW_DATA_FILE, DAILY_DATA_FILE,
    TOMTOM_FLOW_URL, OPENWEATHER_URL,
    BANGALORE_CENTER_LAT, BANGALORE_CENTER_LON,
    COLLECTION_INTERVAL_MINUTES,
)

try:
    import requests
except ImportError:
    st.error("Install requests: pip install requests")
    st.stop()

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Ensure data/logs directories exist
os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)

# PID file for background collector
PID_FILE = os.path.join(SCRIPT_DIR, "logs", "collector.pid")
os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)

# ============================================================================
# RAW CSV HEADERS (same as realtime_data_collector.py)
# ============================================================================
RAW_HEADERS = [
    'timestamp', 'date', 'time', 'hour', 'day_of_week',
    'road_name', 'area_name', 'latitude', 'longitude',
    'tomtom_current_speed', 'tomtom_free_flow_speed',
    'tomtom_current_travel_time', 'tomtom_free_flow_travel_time',
    'tomtom_confidence', 'tomtom_road_closure',
    'here_speed', 'here_free_flow_speed', 'here_jam_factor', 'here_traversability',
    'weather_main', 'weather_description', 'temperature_c', 'humidity_pct',
    'wind_speed_ms', 'visibility_m', 'rain_1h_mm',
    'road_lanes', 'road_type', 'max_capacity_vph',
    'incident_count', 'has_construction',
    'collection_status',
]

# ============================================================================
# DISPLAY NAME MAPPING (raw CSV names -> clean readable names)
# ============================================================================
DISPLAY_NAMES = {
    'timestamp': 'Timestamp',
    'date': 'Date',
    'time': 'Time',
    'hour': 'Hour',
    'day_of_week': 'Day of Week',
    'road_name': 'Road Name',
    'area_name': 'Area Name',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'tomtom_current_speed': 'Current Speed (km/h)',
    'tomtom_free_flow_speed': 'Free Flow Speed (km/h)',
    'tomtom_current_travel_time': 'Travel Time (s)',
    'tomtom_free_flow_travel_time': 'Free Flow Travel Time (s)',
    'tomtom_confidence': 'Confidence',
    'tomtom_road_closure': 'Road Closure',
    'here_speed': 'Speed - HERE (km/h)',
    'here_free_flow_speed': 'Free Flow - HERE (km/h)',
    'here_jam_factor': 'Jam Factor (0-10)',
    'here_traversability': 'Traversability',
    'weather_main': 'Weather',
    'weather_description': 'Weather Detail',
    'temperature_c': 'Temperature (C)',
    'humidity_pct': 'Humidity (%)',
    'wind_speed_ms': 'Wind Speed (m/s)',
    'visibility_m': 'Visibility (m)',
    'rain_1h_mm': 'Rain (mm/h)',
    'road_lanes': 'Lanes',
    'road_type': 'Road Type',
    'max_capacity_vph': 'Max Capacity (vph)',
    'incident_count': 'Incidents',
    'has_construction': 'Construction',
    'collection_status': 'Status',
}


def rename_for_display(df):
    """Rename raw column names to clean display names for user-facing tables."""
    return df.rename(columns={k: v for k, v in DISPLAY_NAMES.items() if k in df.columns})


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bangalore Traffic - Real-Time Data Collection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 5px; }
    .collection-active {
        background: #d4edda; border: 2px solid #28a745; border-radius: 10px;
        padding: 10px; text-align: center; color: #155724;
    }
    .collection-inactive {
        background: #f8d7da; border: 2px solid #dc3545; border-radius: 10px;
        padding: 10px; text-align: center; color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API FUNCTIONS
# ============================================================================

def fetch_tomtom_flow(lat, lon):
    """Fetch real-time traffic flow from TomTom."""
    params = {
        'point': f'{lat},{lon}',
        'unit': 'KMPH',
        'thickness': 1,
        'key': TOMTOM_API_KEY,
    }
    try:
        resp = requests.get(TOMTOM_FLOW_URL, params=params, timeout=10)
        resp.raise_for_status()
        flow = resp.json().get('flowSegmentData', {})
        return {
            'current_speed': flow.get('currentSpeed'),
            'free_flow_speed': flow.get('freeFlowSpeed'),
            'current_travel_time': flow.get('currentTravelTime'),
            'free_flow_travel_time': flow.get('freeFlowTravelTime'),
            'confidence': flow.get('confidence'),
            'road_closure': flow.get('roadClosure', False),
            'status': 'success'
        }
    except Exception as e:
        return {'current_speed': None, 'free_flow_speed': None, 'confidence': None,
                'current_travel_time': None, 'free_flow_travel_time': None,
                'road_closure': None, 'status': f'error: {e}'}


def fetch_weather():
    """Fetch weather for Bangalore."""
    params = {
        'lat': BANGALORE_CENTER_LAT,
        'lon': BANGALORE_CENTER_LON,
        'units': 'metric',
        'appid': OPENWEATHER_API_KEY,
    }
    try:
        resp = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        weather_info = data.get('weather', [{}])[0]
        main_data = data.get('main', {})
        wind = data.get('wind', {})
        rain = data.get('rain', {}).get('1h', 0)
        return {
            'weather_main': weather_info.get('main', 'Unknown'),
            'description': weather_info.get('description', ''),
            'temperature': main_data.get('temp'),
            'humidity': main_data.get('humidity'),
            'wind_speed': wind.get('speed'),
            'visibility': data.get('visibility'),
            'rain_1h': rain if rain else 0,
            'status': 'success'
        }
    except Exception as e:
        return {'weather_main': None, 'temperature': None, 'humidity': None,
                'wind_speed': None, 'visibility': None, 'rain_1h': 0,
                'description': '', 'status': f'error: {e}'}


def fetch_tomtom_incidents(lat, lon):
    """Fetch traffic incidents near a location."""
    delta = 500 / 111000
    bbox = f'{lat-delta},{lon-delta},{lat+delta},{lon+delta}'
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        'bbox': bbox,
        'fields': '{incidents{type,properties{delay,magnitudeOfDelay}}}',
        'language': 'en-GB',
        'categoryFilter': '0,1,2,3,4,5,6,7,8,9,10,11,14',
        'key': TOMTOM_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        incidents = resp.json().get('incidents', [])
        has_construction = any(
            inc.get('type', '') in ('CONSTRUCTION', 'ROAD_WORK', 'ROAD_CLOSURE')
            for inc in incidents
        )
        return {'incident_count': len(incidents), 'has_construction': has_construction}
    except Exception:
        return {'incident_count': 0, 'has_construction': False}


# ============================================================================
# CORE: FETCH + SAVE TO CSV
# ============================================================================

def collect_and_save():
    """
    Fetch live data for ALL 16 roads + weather, then SAVE to raw_traffic_data.csv.
    Returns (road_results_list, weather_dict, records_saved_count).
    """
    now = datetime.now(IST)

    # 1. Fetch weather once for Bangalore
    weather = fetch_weather()

    # 2. Fetch incidents only on the hour
    fetch_incidents = (now.minute < 15)

    # 3. Fetch traffic for each road + build CSV rows
    road_results = []
    csv_rows = []
    success_count = 0

    for road in ROAD_LOCATIONS:
        lat, lon = road['lat'], road['lon']
        tt = fetch_tomtom_flow(lat, lon)
        time.sleep(0.25)

        # Derive jam factor from TomTom
        tt_speed = tt.get('current_speed')
        tt_ff = tt.get('free_flow_speed')
        if tt_speed is not None and tt_ff is not None and tt_ff > 0:
            speed_ratio = min(tt_speed / tt_ff, 1.0)
            jam_factor = round((1 - speed_ratio) * 10, 2)
            congestion_pct = round((1 - speed_ratio) * 100, 1)
        else:
            jam_factor = None
            congestion_pct = None

        # Incidents
        if fetch_incidents:
            inc = fetch_tomtom_incidents(lat, lon)
            time.sleep(0.15)
        else:
            inc = {'incident_count': 0, 'has_construction': False}

        # Overall status
        statuses = [tt['status'], weather['status']]
        if all(s == 'success' for s in statuses):
            overall = 'success'
            success_count += 1
        elif any(s == 'success' for s in statuses):
            overall = 'partial'
            success_count += 1
        else:
            overall = 'failed'

        # CSV row (matches RAW_HEADERS exactly)
        row = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'road_name': road['road'],
            'area_name': road['area'],
            'latitude': lat,
            'longitude': lon,
            'tomtom_current_speed': tt_speed,
            'tomtom_free_flow_speed': tt_ff,
            'tomtom_current_travel_time': tt.get('current_travel_time'),
            'tomtom_free_flow_travel_time': tt.get('free_flow_travel_time'),
            'tomtom_confidence': tt.get('confidence'),
            'tomtom_road_closure': tt.get('road_closure'),
            'here_speed': tt_speed,
            'here_free_flow_speed': tt_ff,
            'here_jam_factor': jam_factor,
            'here_traversability': 'closed' if tt.get('road_closure') else 'open',
            'weather_main': weather.get('weather_main'),
            'weather_description': weather.get('description'),
            'temperature_c': weather.get('temperature'),
            'humidity_pct': weather.get('humidity'),
            'wind_speed_ms': weather.get('wind_speed'),
            'visibility_m': weather.get('visibility'),
            'rain_1h_mm': weather.get('rain_1h', 0),
            'road_lanes': road['lanes'],
            'road_type': road['road_type'],
            'max_capacity_vph': road['max_capacity_vph'],
            'incident_count': inc.get('incident_count', 0),
            'has_construction': inc.get('has_construction', False),
            'collection_status': overall,
        }
        csv_rows.append(row)

        # For dashboard display
        road_results.append({
            'road': road['road'],
            'area': road['area'],
            'lat': lat, 'lon': lon,
            'lanes': road['lanes'],
            'road_type': road['road_type'],
            'current_speed': tt_speed,
            'free_flow_speed': tt_ff,
            'travel_time': tt.get('current_travel_time'),
            'ff_travel_time': tt.get('free_flow_travel_time'),
            'confidence': tt.get('confidence'),
            'road_closure': tt.get('road_closure'),
            'jam_factor': jam_factor,
            'congestion_pct': congestion_pct,
            'incidents': inc.get('incident_count', 0),
            'status': tt['status'],
        })

    # 4. SAVE TO CSV
    file_exists = os.path.exists(RAW_DATA_FILE) and os.path.getsize(RAW_DATA_FILE) > 0
    with open(RAW_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_HEADERS)
        if not file_exists:
            writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return road_results, weather, success_count


def get_congestion_color(congestion_pct):
    if congestion_pct is None:
        return '#95a5a6'
    if congestion_pct < 20:
        return '#27ae60'
    elif congestion_pct < 40:
        return '#f1c40f'
    elif congestion_pct < 60:
        return '#e67e22'
    elif congestion_pct < 80:
        return '#e74c3c'
    else:
        return '#8e44ad'


def get_congestion_label(congestion_pct):
    if congestion_pct is None:
        return 'Unknown'
    if congestion_pct < 20:
        return 'Free Flow'
    elif congestion_pct < 40:
        return 'Light'
    elif congestion_pct < 60:
        return 'Moderate'
    elif congestion_pct < 80:
        return 'Heavy'
    else:
        return 'Severe'


def load_raw_data():
    """Load raw CSV if it exists."""
    if os.path.exists(RAW_DATA_FILE) and os.path.getsize(RAW_DATA_FILE) > 0:
        return pd.read_csv(RAW_DATA_FILE)
    return None


def is_collector_running():
    """Check if background collector process is running."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists (Windows-compatible)
            result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                                    capture_output=True, text=True, timeout=5)
            return str(pid) in result.stdout
        except Exception:
            return False
    return False


def start_background_collector():
    """Start the continuous collector as a background process."""
    collector_script = os.path.join(SCRIPT_DIR, "realtime_data_collector.py")
    log_file = os.path.join(SCRIPT_DIR, "logs", "collection.log")
    try:
        proc = subprocess.Popen(
            [sys.executable, collector_script, '--continuous'],
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            cwd=SCRIPT_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        with open(PID_FILE, 'w') as f:
            f.write(str(proc.pid))
        return proc.pid
    except Exception as e:
        return None


def stop_background_collector():
    """Stop the background collector process."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                          capture_output=True, timeout=10)
            os.remove(PID_FILE)
            return True
        except Exception:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            return False
    return False


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("Traffic Data Dashboard")
st.sidebar.markdown("**PhD Research - Bangalore Traffic Flow Prediction**")
st.sidebar.markdown("---")

# API Status
st.sidebar.subheader("API Status")
tomtom_ok = TOMTOM_API_KEY and TOMTOM_API_KEY != "YOUR_TOMTOM_API_KEY_HERE"
weather_ok = OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_OPENWEATHER_API_KEY_HERE"
st.sidebar.markdown(f"TomTom: {'**Active**' if tomtom_ok else '**Not Configured**'}")
st.sidebar.markdown(f"OpenWeatherMap: {'**Active**' if weather_ok else '**Not Configured**'}")

st.sidebar.markdown("---")

# Background Collector Status
st.sidebar.subheader("Auto-Collection (15 min)")
collector_running = is_collector_running()

if collector_running:
    st.sidebar.markdown('<div class="collection-active"><b>RUNNING</b> - Collecting every 15 min</div>',
                       unsafe_allow_html=True)
    if st.sidebar.button("Stop Auto-Collection", type="secondary"):
        stop_background_collector()
        st.rerun()
else:
    st.sidebar.markdown('<div class="collection-inactive"><b>STOPPED</b> - Not collecting</div>',
                       unsafe_allow_html=True)
    if st.sidebar.button("Start Auto-Collection", type="primary"):
        pid = start_background_collector()
        if pid:
            st.sidebar.success(f"Started! PID: {pid}")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Failed to start. Run manually:\n`python realtime_data_collector.py --continuous`")

st.sidebar.markdown("---")

# Collection Stats
st.sidebar.subheader("Collection Stats")
raw_df_sidebar = load_raw_data()
if raw_df_sidebar is not None:
    st.sidebar.markdown(f"Total records: **{len(raw_df_sidebar):,}**")
    st.sidebar.markdown(f"Collections: **{raw_df_sidebar['timestamp'].nunique()}**")
    st.sidebar.markdown(f"Date range: **{raw_df_sidebar['date'].min()}** to **{raw_df_sidebar['date'].max()}**")
    st.sidebar.markdown(f"Days: **{raw_df_sidebar['date'].nunique()}**")
    last_ts = raw_df_sidebar['timestamp'].max()
    st.sidebar.markdown(f"Last collection: **{last_ts}**")
else:
    st.sidebar.markdown("No data collected yet.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"Roads: **16** | Areas: **8**")
st.sidebar.markdown(f"Interval: **Every {COLLECTION_INTERVAL_MINUTES} min**")
st.sidebar.markdown(f"Daily: **96 collections**")
st.sidebar.markdown(f"Storage: `data/raw_traffic_data.csv`")


# ============================================================================
# TABS
# ============================================================================
st.title("Bangalore Real-Time Traffic Data Collection")
st.markdown("*Every fetch is automatically saved to CSV for your research*")

tabs = st.tabs([
    "Live Dashboard",
    "Road Map",
    "Stored Data",
    "Analytics",
    "Collection Log",
    "How It Works",
])


# ============================================================================
# TAB 1: LIVE DASHBOARD (FETCH + SAVE)
# ============================================================================
with tabs[0]:
    st.header("Live Traffic Data")
    st.markdown("**Every click fetches data from APIs AND saves it to `raw_traffic_data.csv`**")

    col_btn1, col_btn2, col_time = st.columns([1.5, 1.5, 3])
    with col_btn1:
        fetch_btn = st.button("Fetch & Save Now", type="primary", use_container_width=True)
    with col_btn2:
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    with col_time:
        now = datetime.now(IST)
        st.markdown(f"**Current Time (IST):** {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(60)
        st.rerun()

    if fetch_btn:
        with st.spinner("Fetching from TomTom + OpenWeatherMap for 16 roads and SAVING to CSV..."):
            road_results, weather, success_count = collect_and_save()

        st.success(f"**Saved {len(road_results)} records to `{RAW_DATA_FILE}`** | {success_count}/16 roads successful")

        # Weather display
        if weather['status'] == 'success':
            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            wc1.metric("Weather", weather['weather_main'])
            wc2.metric("Temperature", f"{weather['temperature']}C")
            wc3.metric("Humidity", f"{weather['humidity']}%")
            wc4.metric("Wind", f"{weather['wind_speed']} m/s")
            wc5.metric("Rain (1h)", f"{weather['rain_1h']} mm")
        else:
            st.warning(f"Weather API: {weather['status']}")

        st.markdown("---")

        df_live = pd.DataFrame(road_results)
        successful = df_live[df_live['status'] == 'success']

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Roads Fetched", f"{len(successful)}/16")
        if len(successful) > 0:
            mc2.metric("Avg Speed", f"{successful['current_speed'].mean():.1f} km/h")
            mc3.metric("Avg Congestion", f"{successful['congestion_pct'].mean():.1f}%")
            mc4.metric("Avg Free Flow", f"{successful['free_flow_speed'].mean():.1f} km/h")

        st.markdown("---")

        # Road-wise table
        st.subheader("Road-wise Live Traffic")
        display_rows = []
        for _, row in df_live.iterrows():
            if row['status'] == 'success':
                display_rows.append({
                    'Road': row['road'],
                    'Area': row['area'],
                    'Current Speed (km/h)': row['current_speed'],
                    'Free Flow (km/h)': row['free_flow_speed'],
                    'Congestion %': row['congestion_pct'],
                    'Jam Factor': row['jam_factor'],
                    'Status': get_congestion_label(row['congestion_pct']),
                    'Confidence': row['confidence'],
                    'Incidents': row['incidents'],
                })
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Speed comparison chart
        st.subheader("Current Speed vs Free Flow Speed")
        if len(successful) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=successful['road'], y=successful['free_flow_speed'],
                name='Free Flow Speed', marker_color='#3498db', opacity=0.6
            ))
            fig.add_trace(go.Bar(
                x=successful['road'], y=successful['current_speed'],
                name='Current Speed', marker_color='#e74c3c'
            ))
            fig.update_layout(barmode='overlay', height=450, xaxis_tickangle=-45,
                              yaxis_title="Speed (km/h)")
            st.plotly_chart(fig, use_container_width=True)

        # Congestion chart
        st.subheader("Congestion Level by Road")
        if len(successful) > 0:
            colors = [get_congestion_color(c) for c in successful['congestion_pct']]
            fig = go.Figure(go.Bar(
                x=successful['road'], y=successful['congestion_pct'],
                marker_color=colors,
                text=[f"{c:.0f}%" for c in successful['congestion_pct']],
                textposition='outside',
            ))
            fig.update_layout(height=400, xaxis_tickangle=-45, yaxis_title="Congestion %",
                              yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)

        # Store in session for map tab
        st.session_state['live_data'] = df_live
        st.session_state['live_weather'] = weather

    else:
        # Show last collection from CSV
        raw_df = load_raw_data()
        if raw_df is not None and len(raw_df) > 0:
            last_ts = raw_df['timestamp'].max()
            last_data = raw_df[raw_df['timestamp'] == last_ts]
            st.info(f"**Last saved collection:** {last_ts} IST ({len(last_data)} roads)")

            st.subheader("Last Saved Data")
            cols_to_show = ['road_name', 'area_name', 'tomtom_current_speed', 'tomtom_free_flow_speed',
                           'here_jam_factor', 'weather_main', 'temperature_c', 'collection_status']
            avail = [c for c in cols_to_show if c in last_data.columns]
            st.dataframe(rename_for_display(last_data[avail].reset_index(drop=True)), use_container_width=True, hide_index=True)
        else:
            st.info("Click **'Fetch & Save Now'** to collect and store real-time data for all 16 roads.")


# ============================================================================
# TAB 2: ROAD MAP
# ============================================================================
with tabs[1]:
    st.header("Bangalore Road Locations Map")

    if HAS_FOLIUM:
        live_data = st.session_state.get('live_data', None)
        m = folium.Map(location=[12.9716, 77.6200], zoom_start=12, tiles='OpenStreetMap')

        for road in ROAD_LOCATIONS:
            speed_info = ""
            color = 'blue'
            if live_data is not None:
                match = live_data[live_data['road'] == road['road']]
                if len(match) > 0 and match.iloc[0]['status'] == 'success':
                    r = match.iloc[0]
                    speed_info = (f"<br><b>Speed: {r['current_speed']} km/h</b>"
                                  f"<br>Free Flow: {r['free_flow_speed']} km/h"
                                  f"<br>Congestion: {r['congestion_pct']}%")
                    color = 'green' if r['congestion_pct'] < 30 else 'orange' if r['congestion_pct'] < 60 else 'red'

            popup_html = (f"<b>{road['road']}</b><br>Area: {road['area']}<br>"
                         f"Type: {road['road_type']}<br>Lanes: {road['lanes']}<br>"
                         f"Max Capacity: {road['max_capacity_vph']} vph{speed_info}")

            folium.Marker(
                location=[road['lat'], road['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{road['road']} ({road['area']})",
                icon=folium.Icon(color=color, icon='road', prefix='fa'),
            ).add_to(m)

        st_folium(m, width=None, height=600)
        st.markdown("**Legend:** Green = Free Flow | Orange = Moderate | Red = Heavy | Blue = No data yet")

        st.subheader("Road Details")
        road_df = pd.DataFrame(ROAD_LOCATIONS)
        road_df.columns = ['Road', 'Area', 'Lat', 'Lon', 'Length (km)', 'Lanes', 'Type', 'Max Capacity']
        st.dataframe(road_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Install folium: `pip install folium streamlit-folium`")


# ============================================================================
# TAB 3: STORED DATA
# ============================================================================
with tabs[2]:
    st.header("Stored Data Explorer")

    raw_df = load_raw_data()

    if raw_df is not None:
        # Summary
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Total Records", f"{len(raw_df):,}")
        mc2.metric("Collections", raw_df['timestamp'].nunique())
        mc3.metric("Days", raw_df['date'].nunique())
        mc4.metric("Roads", raw_df['road_name'].nunique())
        success_pct = (raw_df['collection_status'] == 'success').sum() / len(raw_df) * 100
        mc5.metric("Success Rate", f"{success_pct:.1f}%")

        st.markdown("---")

        # Data size info
        file_size_mb = os.path.getsize(RAW_DATA_FILE) / (1024 * 1024)
        st.markdown(f"**File:** `{RAW_DATA_FILE}` | **Size:** {file_size_mb:.2f} MB | "
                   f"**Records per collection:** 16 roads")

        st.markdown("---")

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            dates = sorted(raw_df['date'].unique(), reverse=True)
            selected_date = st.selectbox("Date", ['All'] + dates, key="sd_date")
        with col_f2:
            areas = sorted(raw_df['area_name'].unique())
            selected_area = st.selectbox("Area", ['All'] + list(areas), key="sd_area")
        with col_f3:
            roads = sorted(raw_df['road_name'].unique())
            selected_road = st.selectbox("Road", ['All'] + list(roads), key="sd_road")

        filtered = raw_df.copy()
        if selected_date != 'All':
            filtered = filtered[filtered['date'] == selected_date]
        if selected_area != 'All':
            filtered = filtered[filtered['area_name'] == selected_area]
        if selected_road != 'All':
            filtered = filtered[filtered['road_name'] == selected_road]

        st.subheader(f"Data ({len(filtered):,} records)")
        st.dataframe(rename_for_display(filtered), use_container_width=True, height=400)

        # Statistics
        st.subheader("Statistics")
        num_cols = ['tomtom_current_speed', 'tomtom_free_flow_speed', 'here_jam_factor',
                   'temperature_c', 'humidity_pct']
        avail = [c for c in num_cols if c in filtered.columns]
        if avail:
            st.dataframe(rename_for_display(filtered[avail].describe().round(2)), use_container_width=True)

        # Download
        csv_data = rename_for_display(filtered).to_csv(index=False)
        st.download_button("Download Filtered Data (CSV)", csv_data,
                          "traffic_data_filtered.csv", "text/csv")

        # Collection history
        st.markdown("---")
        st.subheader("Collection History")
        collections = raw_df.groupby('timestamp').agg(
            roads=('road_name', 'count'),
            avg_speed=('tomtom_current_speed', 'mean'),
            success=('collection_status', lambda x: (x == 'success').sum()),
        ).reset_index()
        collections.columns = ['Timestamp', 'Roads', 'Avg Speed (km/h)', 'Successful']
        collections['Avg Speed (km/h)'] = collections['Avg Speed (km/h)'].round(1)
        st.dataframe(collections.sort_values('Timestamp', ascending=False),
                     use_container_width=True, hide_index=True, height=300)

    else:
        st.info("No data stored yet. Go to **Live Dashboard** and click **Fetch & Save Now**.")


# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================
with tabs[3]:
    st.header("Traffic Analytics")

    raw_df = load_raw_data()
    if raw_df is not None and len(raw_df) > 0:

        # Speed over time
        st.subheader("Average Speed Over Time (All Roads)")
        speed_ts = raw_df.groupby('timestamp').agg({
            'tomtom_current_speed': 'mean',
            'tomtom_free_flow_speed': 'mean',
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=speed_ts['timestamp'], y=speed_ts['tomtom_free_flow_speed'],
                                 mode='lines', name='Free Flow Speed', line=dict(color='#3498db', width=1.5)))
        fig.add_trace(go.Scatter(x=speed_ts['timestamp'], y=speed_ts['tomtom_current_speed'],
                                 mode='lines+markers', name='Current Speed', line=dict(color='#e74c3c', width=2),
                                 fill='tonexty', fillcolor='rgba(231,76,60,0.1)'))
        fig.update_layout(height=400, yaxis_title="Speed (km/h)", xaxis_title="Collection Time")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        # Speed by road
        with col_a:
            st.subheader("Average Speed by Road")
            road_speed = raw_df.groupby('road_name')['tomtom_current_speed'].mean().sort_values()
            colors = ['#e74c3c' if s < 20 else '#f39c12' if s < 30 else '#27ae60'
                     for s in road_speed.values]
            fig = go.Figure(go.Bar(
                y=road_speed.index, x=road_speed.values.round(1), orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}" for v in road_speed.values],
                textposition='outside',
            ))
            fig.update_layout(height=500, xaxis_title="Avg Speed (km/h)")
            st.plotly_chart(fig, use_container_width=True)

        # Congestion by area
        with col_b:
            st.subheader("Congestion by Area (Jam Factor)")
            if 'here_jam_factor' in raw_df.columns:
                area_jam = raw_df.groupby('area_name')['here_jam_factor'].mean().sort_values(ascending=False)
                colors = [get_congestion_color(j * 10) for j in area_jam.values]
                fig = go.Figure(go.Bar(
                    x=area_jam.index, y=area_jam.values.round(2),
                    marker_color=colors,
                    text=[f"{v:.1f}/10" for v in area_jam.values],
                    textposition='outside',
                ))
                fig.update_layout(height=500, yaxis_title="Avg Jam Factor (0-10)")
                st.plotly_chart(fig, use_container_width=True)

        # Hourly pattern (if multiple hours collected)
        if raw_df['hour'].nunique() > 1:
            st.subheader("Hourly Traffic Speed Pattern")
            hourly = raw_df.groupby('hour')['tomtom_current_speed'].mean().reset_index()
            fig = go.Figure(go.Scatter(
                x=hourly['hour'], y=hourly['tomtom_current_speed'].round(1),
                mode='lines+markers', line=dict(color='#8e44ad', width=3),
                fill='tozeroy', fillcolor='rgba(142,68,173,0.1)',
            ))
            fig.update_layout(height=350, xaxis_title="Hour of Day", yaxis_title="Avg Speed (km/h)",
                              xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig, use_container_width=True)

        # Weather impact
        if 'weather_main' in raw_df.columns and raw_df['weather_main'].notna().any():
            st.subheader("Weather Impact on Speed")
            ws = raw_df.groupby('weather_main')['tomtom_current_speed'].agg(['mean', 'count']).reset_index()
            ws.columns = ['Weather', 'Avg Speed', 'Count']
            ws = ws[ws['Count'] >= 5].sort_values('Avg Speed')
            if len(ws) > 0:
                fig = go.Figure(go.Bar(x=ws['Weather'], y=ws['Avg Speed'].round(1), marker_color='#e67e22',
                                       text=[f"{v:.1f} km/h" for v in ws['Avg Speed']], textposition='outside'))
                fig.update_layout(height=350, yaxis_title="Avg Speed (km/h)")
                st.plotly_chart(fig, use_container_width=True)

        # Collection timeline
        st.subheader("Records Collected Per Day")
        daily_counts = raw_df.groupby('date').size().reset_index(name='Records')
        fig = go.Figure(go.Bar(x=daily_counts['date'], y=daily_counts['Records'], marker_color='#2ecc71'))
        fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No data to analyze. Collect some data first!")


# ============================================================================
# TAB 5: COLLECTION LOG
# ============================================================================
with tabs[4]:
    st.header("Collection Log & Status")

    log_file = os.path.join(SCRIPT_DIR, "logs", "collection.log")

    # Status card
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Background Collector")
        if is_collector_running():
            st.success("**Status: RUNNING** - Auto-collecting every 15 minutes")
        else:
            st.error("**Status: STOPPED** - Start from sidebar or run:\n`python realtime_data_collector.py --continuous`")

    with col2:
        st.subheader("Data File Status")
        if os.path.exists(RAW_DATA_FILE):
            size = os.path.getsize(RAW_DATA_FILE)
            st.info(f"**File:** `raw_traffic_data.csv`\n\n"
                   f"**Size:** {size / 1024:.1f} KB ({size / (1024*1024):.2f} MB)\n\n"
                   f"**Path:** `{RAW_DATA_FILE}`")
        else:
            st.warning("No data file yet.")

    st.markdown("---")

    # Collection summary
    raw_df = load_raw_data()
    if raw_df is not None:
        st.subheader("Collection Summary")
        total_collections = raw_df['timestamp'].nunique()
        total_records = len(raw_df)
        days = raw_df['date'].nunique()
        first = raw_df['timestamp'].min()
        last = raw_df['timestamp'].max()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Collections", total_collections)
        sc2.metric("Total Records", f"{total_records:,}")
        sc3.metric("Days Active", days)
        sc4.metric("Records/Collection", f"{total_records // max(total_collections, 1)}")

        st.markdown(f"**First collection:** {first} | **Last collection:** {last}")

        # Success rate by road
        st.subheader("Success Rate by Road")
        road_success = raw_df.groupby('road_name')['collection_status'].apply(
            lambda x: (x == 'success').sum() / len(x) * 100
        ).sort_values()
        fig = go.Figure(go.Bar(
            y=road_success.index, x=road_success.values.round(1), orientation='h',
            marker_color=['#27ae60' if v > 90 else '#f39c12' if v > 70 else '#e74c3c'
                         for v in road_success.values],
            text=[f"{v:.1f}%" for v in road_success.values],
            textposition='outside',
        ))
        fig.update_layout(height=500, xaxis_title="Success Rate %", xaxis=dict(range=[0, 105]))
        st.plotly_chart(fig, use_container_width=True)

    # Log viewer
    st.markdown("---")
    st.subheader("Recent Logs")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        last_lines = lines[-50:] if len(lines) > 50 else lines
        st.code(''.join(last_lines), language='log')
    else:
        st.info("No logs yet. Start a collection to see logs.")


# ============================================================================
# TAB 6: HOW IT WORKS
# ============================================================================
with tabs[5]:
    st.header("How the Real-Time Data Collection Works")

    st.markdown("""
    ### Research Objective
    Collect real-time traffic data for **16 major Bangalore roads** every **15 minutes** for
    **3 months** to build and validate machine learning models for traffic flow prediction.
    """)

    st.markdown("---")

    st.subheader("Data Collection Pipeline")
    st.markdown("""
    ```
    +------------------+     +------------------+     +-------------------+
    |   TomTom API     |     | OpenWeatherMap   |     |   TomTom API      |
    |  Traffic Flow    |     |   Weather API    |     |   Incidents       |
    +--------+---------+     +--------+---------+     +--------+----------+
             |                        |                        |
             v                        v                        v
    +--------+---------+     +--------+---------+     +--------+----------+
    | Current Speed    |     | Temperature      |     | Accident Count    |
    | Free Flow Speed  |     | Humidity         |     | Roadwork Status   |
    | Travel Time      |     | Rain / Wind      |     | Road Closures     |
    | Confidence Score |     | Visibility       |     |                   |
    +--------+---------+     +--------+---------+     +--------+----------+
             |                        |                        |
             +------------------------+------------------------+
                                      |
                                      v
                    +----------------------------------+
                    |  raw_traffic_data.csv             |
                    |  PERMANENTLY STORED               |
                    |  Every fetch = 16 new rows saved  |
                    |  ~138,240 records over 3 months   |
                    +----------------------------------+
                                      |
                                      v
                    +----------------------------------+
                    |  preprocess_realtime_data.py      |
                    |  Converts to daily format         |
                    |  Real API-sourced columns only    |
                    +----------------------------------+
    ```
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### TomTom Traffic Flow API
        - **Endpoint:** Flow Segment Data v4
        - **Data:** Real-time speed, free-flow speed, travel time
        - **Update frequency:** Every 1 minute
        - **Free tier:** 2,500 requests/day
        - **Our usage:** ~1,536 requests/day
        - **Cost: FREE**

        **Sample Response:**
        ```json
        {
          "flowSegmentData": {
            "currentSpeed": 17,
            "freeFlowSpeed": 22,
            "currentTravelTime": 42,
            "freeFlowTravelTime": 32,
            "confidence": 1.0,
            "roadClosure": false
          }
        }
        ```
        """)

    with col2:
        st.markdown("""
        #### OpenWeatherMap API (Free v2.5)
        - **Endpoint:** Current Weather API
        - **Data:** Temperature, humidity, rain, wind
        - **Update frequency:** Every 10 minutes
        - **Free tier:** 1,000 requests/day
        - **Our usage:** ~96 requests/day
        - **Cost: FREE**

        **Sample Response:**
        ```json
        {
          "weather": [{"main": "Clear"}],
          "main": {"temp": 28.5, "humidity": 65},
          "wind": {"speed": 3.2},
          "visibility": 10000
        }
        ```
        """)

    st.markdown("---")

    st.subheader("Derived Metrics")
    st.markdown("""
    | Metric | Formula | Description |
    |--------|---------|-------------|
    | **Jam Factor** | `(1 - current_speed / free_flow_speed) x 10` | 0 = free flow, 10 = standstill |
    | **Congestion %** | `(1 - current_speed / free_flow_speed) x 100` | Percentage of congestion |
    | **Travel Time Index** | `current_travel_time / free_flow_travel_time` | >1 means delay |
    | **Traffic Volume** | Greenshields Model | Estimated from speed-flow relationship |
    """)

    st.markdown("---")

    st.subheader("3-Month Cost: $0")
    cost_data = {
        'API': ['TomTom Traffic Flow', 'TomTom Incidents', 'OpenWeatherMap'],
        'Daily Calls': ['1,536', '~96', '96'],
        'Free Limit': ['2,500/day', 'Shared', '1,000/day'],
        '3-Month Cost': ['$0', '$0', '$0'],
    }
    st.table(pd.DataFrame(cost_data))

    st.markdown("---")

    st.subheader("16 Monitored Roads")
    road_df = pd.DataFrame(ROAD_LOCATIONS)
    road_df.columns = ['Road', 'Area', 'Lat', 'Lon', 'Length (km)', 'Lanes', 'Type', 'Max Capacity']
    st.dataframe(road_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Output Columns (Real API-Sourced)")
    st.markdown("""
    | Column | Source |
    |--------|--------|
    | Date | Collection timestamp |
    | Area Name | Config (8 areas) |
    | Road/Intersection Name | Config (16 roads) |
    | Traffic Volume | Greenshields Model |
    | Average Speed | TomTom current_speed |
    | Travel Time Index | travel_time / free_flow_travel_time |
    | Congestion Level | Derived from speed ratio |
    | Road Capacity Utilization | Derived from speed ratio |
    | Incident Reports | TomTom Incidents API |
    | Weather Conditions | OpenWeatherMap |
    | Roadwork and Construction | TomTom Incidents |
    | Temperature | OpenWeatherMap |
    | Humidity | OpenWeatherMap |
    | Wind Speed | OpenWeatherMap |
    | Rain Volume | OpenWeatherMap |
    | Visibility | OpenWeatherMap |
    | Free Flow Speed | TomTom API |
    | Jam Factor | TomTom (derived) |
    | Confidence | TomTom API |
    """)
