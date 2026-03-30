"""
=============================================================================
REAL-TIME TRAFFIC DATA COLLECTOR FOR BANGALORE
=============================================================================
Collects traffic data from TomTom, HERE, and OpenWeatherMap APIs
for 16 Bangalore roads every 15 minutes.

Usage:
  python realtime_data_collector.py              # Single collection run
  python realtime_data_collector.py --continuous  # Run continuously (every 15 min)
  python realtime_data_collector.py --test        # Test API connections only

Author: PhD Research - Traffic Flow Prediction
=============================================================================
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    print("ERROR: 'requests' module not found. Install it with:")
    print("  pip install requests")
    sys.exit(1)

from config import (
    TOMTOM_API_KEY, OPENWEATHER_API_KEY,
    ROAD_LOCATIONS, RAW_DATA_FILE, LOG_FILE,
    TOMTOM_FLOW_URL, OPENWEATHER_URL,
    BANGALORE_CENTER_LAT, BANGALORE_CENTER_LON,
    COLLECTION_INTERVAL_MINUTES
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# ============================================================================
# RAW CSV HEADERS
# ============================================================================
RAW_HEADERS = [
    'timestamp',             # IST datetime
    'date',                  # YYYY-MM-DD
    'time',                  # HH:MM:SS
    'hour',                  # 0-23
    'day_of_week',           # 0=Mon, 6=Sun
    'road_name',             # Road/Intersection Name
    'area_name',             # Area Name
    'latitude',
    'longitude',
    # TomTom data
    'tomtom_current_speed',       # km/h
    'tomtom_free_flow_speed',     # km/h
    'tomtom_current_travel_time', # seconds
    'tomtom_free_flow_travel_time', # seconds
    'tomtom_confidence',          # 0-1
    'tomtom_road_closure',        # True/False
    # HERE data
    'here_speed',                 # km/h
    'here_free_flow_speed',       # km/h
    'here_jam_factor',            # 0.0 - 10.0
    'here_traversability',        # open/closed
    # Weather data
    'weather_main',               # Clear/Rain/Clouds/etc.
    'weather_description',        # detailed description
    'temperature_c',              # Celsius
    'humidity_pct',               # %
    'wind_speed_ms',              # m/s
    'visibility_m',               # meters
    'rain_1h_mm',                 # mm (0 if no rain)
    # Road characteristics (from config)
    'road_lanes',
    'road_type',
    'max_capacity_vph',
    # Incidents (from TomTom)
    'incident_count',             # incidents within 500m
    'has_construction',           # True/False (roadwork nearby)
    # Collection metadata
    'collection_status',          # success/partial/failed
]


# ============================================================================
# API CALL FUNCTIONS
# ============================================================================

def fetch_tomtom_flow(lat, lon, api_key, timeout=10):
    """
    Fetch real-time traffic flow data from TomTom Traffic Flow API.

    Returns: dict with currentSpeed, freeFlowSpeed, currentTravelTime,
             freeFlowTravelTime, confidence, roadClosure
    """
    params = {
        'point': f'{lat},{lon}',
        'unit': 'KMPH',
        'thickness': 1,
        'key': api_key,
    }
    try:
        resp = requests.get(TOMTOM_FLOW_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        flow = data.get('flowSegmentData', {})
        return {
            'tomtom_current_speed': flow.get('currentSpeed', None),
            'tomtom_free_flow_speed': flow.get('freeFlowSpeed', None),
            'tomtom_current_travel_time': flow.get('currentTravelTime', None),
            'tomtom_free_flow_travel_time': flow.get('freeFlowTravelTime', None),
            'tomtom_confidence': flow.get('confidence', None),
            'tomtom_road_closure': flow.get('roadClosure', False),
            'status': 'success'
        }
    except requests.exceptions.RequestException as e:
        logger.warning(f"TomTom Flow API error for ({lat},{lon}): {e}")
        return {
            'tomtom_current_speed': None,
            'tomtom_free_flow_speed': None,
            'tomtom_current_travel_time': None,
            'tomtom_free_flow_travel_time': None,
            'tomtom_confidence': None,
            'tomtom_road_closure': None,
            'status': f'error: {e}'
        }


def fetch_here_flow(lat, lon, api_key, timeout=10):
    """
    Fetch real-time traffic flow data from HERE Traffic API v7.

    Returns: dict with speed, freeFlowSpeed, jamFactor, traversability
    """
    params = {
        'in': f'circle:{lat},{lon};r=100',
        'locationReferencing': 'shape',
        'apiKey': api_key,
    }
    try:
        resp = requests.get(HERE_FLOW_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        results = data.get('results', [])
        if results:
            # Take the first (closest) result
            loc = results[0].get('location', {})
            current_flow = results[0].get('currentFlow', {})

            speed = current_flow.get('speed', None)
            if speed is not None:
                speed = round(speed * 3.6, 2)   # m/s to km/h

            ff_speed = current_flow.get('freeFlow', None)
            if ff_speed is not None:
                ff_speed = round(ff_speed * 3.6, 2)

            return {
                'here_speed': speed,
                'here_free_flow_speed': ff_speed,
                'here_jam_factor': current_flow.get('jamFactor', None),
                'here_traversability': current_flow.get('traversability', 'open'),
                'status': 'success'
            }
        else:
            return {
                'here_speed': None,
                'here_free_flow_speed': None,
                'here_jam_factor': None,
                'here_traversability': None,
                'status': 'no_data'
            }
    except requests.exceptions.RequestException as e:
        logger.warning(f"HERE Flow API error for ({lat},{lon}): {e}")
        return {
            'here_speed': None,
            'here_free_flow_speed': None,
            'here_jam_factor': None,
            'here_traversability': None,
            'status': f'error: {e}'
        }


def fetch_weather(lat, lon, api_key, timeout=10):
    """
    Fetch current weather data from OpenWeatherMap Current Weather API 2.5 (free).

    Returns: dict with weather condition, temperature, humidity, wind, rain
    """
    params = {
        'lat': lat,
        'lon': lon,
        'units': 'metric',
        'appid': api_key,
    }
    try:
        resp = requests.get(OPENWEATHER_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        weather_info = data.get('weather', [{}])[0]
        main = data.get('main', {})
        wind = data.get('wind', {})
        rain = data.get('rain', {}).get('1h', 0)

        return {
            'weather_main': weather_info.get('main', 'Unknown'),
            'weather_description': weather_info.get('description', ''),
            'temperature_c': main.get('temp', None),
            'humidity_pct': main.get('humidity', None),
            'wind_speed_ms': wind.get('speed', None),
            'visibility_m': data.get('visibility', None),
            'rain_1h_mm': rain if rain else 0,
            'status': 'success'
        }
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenWeatherMap API error: {e}")
        return {
            'weather_main': None,
            'weather_description': None,
            'temperature_c': None,
            'humidity_pct': None,
            'wind_speed_ms': None,
            'visibility_m': None,
            'rain_1h_mm': None,
            'status': f'error: {e}'
        }


def fetch_tomtom_incidents(lat, lon, api_key, radius_m=500, timeout=10):
    """
    Fetch traffic incidents near a location from TomTom Incidents API.

    Returns: dict with incident_count and has_construction
    """
    # Bounding box around the point (roughly radius_m)
    delta = radius_m / 111000  # degrees approx
    bbox = f'{lat-delta},{lon-delta},{lat+delta},{lon+delta}'

    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        'bbox': bbox,
        'fields': '{incidents{type,properties{delay,magnitudeOfDelay}}}',
        'language': 'en-GB',
        'categoryFilter': '0,1,2,3,4,5,6,7,8,9,10,11,14',
        'key': api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        incidents = data.get('incidents', [])
        incident_count = len(incidents)

        # Check for construction/roadwork (category 7, 8, 9 in TomTom)
        has_construction = False
        for inc in incidents:
            inc_type = inc.get('type', '')
            if inc_type in ['CONSTRUCTION', 'ROAD_WORK', 'ROAD_CLOSURE']:
                has_construction = True
                break
            # Also check properties
            props = inc.get('properties', {})
            if isinstance(props, dict):
                icon_cat = props.get('iconCategory', 0)
                if icon_cat in [7, 8, 9]:   # roadwork categories
                    has_construction = True
                    break

        return {
            'incident_count': incident_count,
            'has_construction': has_construction,
            'status': 'success'
        }
    except requests.exceptions.RequestException as e:
        logger.warning(f"TomTom Incidents API error for ({lat},{lon}): {e}")
        return {
            'incident_count': 0,
            'has_construction': False,
            'status': f'error: {e}'
        }


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_once():
    """
    Perform one complete data collection cycle for all 16 roads.
    Returns: number of successful collections
    """
    now = datetime.now(IST)
    logger.info(f"=== Collection started at {now.strftime('%Y-%m-%d %H:%M:%S IST')} ===")

    # Fetch weather once for entire Bangalore (same weather across city)
    weather = fetch_weather(BANGALORE_CENTER_LAT, BANGALORE_CENTER_LON,
                           OPENWEATHER_API_KEY)
    if weather['status'] == 'success':
        logger.info(f"Weather: {weather['weather_main']} ({weather['weather_description']}), "
                    f"Temp: {weather['temperature_c']}C, Humidity: {weather['humidity_pct']}%")
    else:
        logger.warning(f"Weather fetch failed: {weather['status']}")

    # Check if CSV exists, if not create with headers
    file_exists = os.path.exists(RAW_DATA_FILE) and os.path.getsize(RAW_DATA_FILE) > 0

    success_count = 0
    rows_to_write = []

    for road_info in ROAD_LOCATIONS:
        road_name = road_info['road']
        area_name = road_info['area']
        lat = road_info['lat']
        lon = road_info['lon']

        # Fetch TomTom flow data
        tomtom = fetch_tomtom_flow(lat, lon, TOMTOM_API_KEY)

        # Small delay to respect rate limits
        time.sleep(0.3)

        # Derive jam factor from TomTom speed data (replaces HERE)
        tt_speed = tomtom.get('tomtom_current_speed')
        tt_ff_speed = tomtom.get('tomtom_free_flow_speed')
        if tt_speed is not None and tt_ff_speed is not None and tt_ff_speed > 0:
            # jam_factor: 0 = free flow, 10 = fully jammed
            speed_ratio = min(tt_speed / tt_ff_speed, 1.0)
            derived_jam_factor = round((1 - speed_ratio) * 10, 2)
        else:
            derived_jam_factor = None

        # Fetch incidents (only every 4th collection to save quota)
        if now.minute == 0:  # Only on the hour
            incidents = fetch_tomtom_incidents(lat, lon, TOMTOM_API_KEY)
            time.sleep(0.2)
        else:
            incidents = {'incident_count': 0, 'has_construction': False, 'status': 'skipped'}

        # Determine overall status
        statuses = [tomtom['status'], weather['status']]
        if all(s == 'success' for s in statuses):
            overall_status = 'success'
            success_count += 1
        elif any(s == 'success' for s in statuses):
            overall_status = 'partial'
            success_count += 1
        else:
            overall_status = 'failed'

        # Build row
        row = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'road_name': road_name,
            'area_name': area_name,
            'latitude': lat,
            'longitude': lon,
            # TomTom
            'tomtom_current_speed': tt_speed,
            'tomtom_free_flow_speed': tt_ff_speed,
            'tomtom_current_travel_time': tomtom.get('tomtom_current_travel_time'),
            'tomtom_free_flow_travel_time': tomtom.get('tomtom_free_flow_travel_time'),
            'tomtom_confidence': tomtom.get('tomtom_confidence'),
            'tomtom_road_closure': tomtom.get('tomtom_road_closure'),
            # Derived from TomTom (replaces HERE)
            'here_speed': tt_speed,  # Use TomTom speed
            'here_free_flow_speed': tt_ff_speed,  # Use TomTom free-flow
            'here_jam_factor': derived_jam_factor,  # Derived from speed ratio
            'here_traversability': 'closed' if tomtom.get('tomtom_road_closure') else 'open',
            # Weather
            'weather_main': weather.get('weather_main'),
            'weather_description': weather.get('weather_description'),
            'temperature_c': weather.get('temperature_c'),
            'humidity_pct': weather.get('humidity_pct'),
            'wind_speed_ms': weather.get('wind_speed_ms'),
            'visibility_m': weather.get('visibility_m'),
            'rain_1h_mm': weather.get('rain_1h_mm'),
            # Road characteristics
            'road_lanes': road_info['lanes'],
            'road_type': road_info['road_type'],
            'max_capacity_vph': road_info['max_capacity_vph'],
            # Incidents
            'incident_count': incidents.get('incident_count', 0),
            'has_construction': incidents.get('has_construction', False),
            # Status
            'collection_status': overall_status,
        }

        rows_to_write.append(row)

        speed_str = f"Speed:{tt_speed or '?'} km/h, Jam:{derived_jam_factor or '?'}"
        logger.info(f"  {road_name:<25} {speed_str:<35} [{overall_status}]")

    # Write all rows to CSV
    with open(RAW_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_HEADERS)
        if not file_exists:
            writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)

    logger.info(f"=== Collection complete: {success_count}/16 roads successful ===")
    logger.info(f"    Data saved to: {RAW_DATA_FILE}")

    return success_count


# ============================================================================
# API CONNECTION TEST
# ============================================================================

def test_apis():
    """Test all API connections with a single road."""
    print("=" * 60)
    print("TESTING API CONNECTIONS")
    print("=" * 60)

    test_road = ROAD_LOCATIONS[0]  # 100 Feet Road
    lat, lon = test_road['lat'], test_road['lon']
    print(f"\nTest location: {test_road['road']} ({lat}, {lon})")

    # Test TomTom
    print(f"\n1. TomTom Traffic Flow API...")
    if TOMTOM_API_KEY == "YOUR_TOMTOM_API_KEY_HERE":
        print("   SKIPPED - API key not configured")
    else:
        result = fetch_tomtom_flow(lat, lon, TOMTOM_API_KEY)
        if result['status'] == 'success':
            print(f"   SUCCESS!")
            print(f"   Current Speed: {result['tomtom_current_speed']} km/h")
            print(f"   Free Flow Speed: {result['tomtom_free_flow_speed']} km/h")
            print(f"   Confidence: {result['tomtom_confidence']}")
        else:
            print(f"   FAILED: {result['status']}")

    # HERE - Skipped (requires payment)
    print(f"\n2. HERE Traffic Flow API...")
    print("   SKIPPED - Using TomTom data instead (HERE requires payment)")

    # Test OpenWeatherMap
    print(f"\n3. OpenWeatherMap API...")
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY_HERE":
        print("   SKIPPED - API key not configured")
    else:
        result = fetch_weather(BANGALORE_CENTER_LAT, BANGALORE_CENTER_LON,
                              OPENWEATHER_API_KEY)
        if result['status'] == 'success':
            print(f"   SUCCESS!")
            print(f"   Weather: {result['weather_main']} ({result['weather_description']})")
            print(f"   Temperature: {result['temperature_c']} C")
            print(f"   Humidity: {result['humidity_pct']}%")
        else:
            print(f"   FAILED: {result['status']}")

    # Test TomTom Incidents
    print(f"\n4. TomTom Incidents API...")
    if TOMTOM_API_KEY == "YOUR_TOMTOM_API_KEY_HERE":
        print("   SKIPPED - API key not configured")
    else:
        result = fetch_tomtom_incidents(lat, lon, TOMTOM_API_KEY)
        if result['status'] == 'success':
            print(f"   SUCCESS!")
            print(f"   Incidents nearby: {result['incident_count']}")
            print(f"   Construction: {result['has_construction']}")
        else:
            print(f"   FAILED: {result['status']}")

    print("\n" + "=" * 60)
    all_configured = all([
        TOMTOM_API_KEY != "YOUR_TOMTOM_API_KEY_HERE",
        OPENWEATHER_API_KEY != "YOUR_OPENWEATHER_API_KEY_HERE",
    ])
    if all_configured:
        print("All APIs configured! Run 'python realtime_data_collector.py' to start.")
    else:
        print("Some API keys are not configured. See API_SETUP_GUIDE.txt")
    print("=" * 60)


# ============================================================================
# CONTINUOUS MODE
# ============================================================================

def run_continuous():
    """Run data collection continuously every 15 minutes."""
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS DATA COLLECTION")
    logger.info(f"Interval: Every {COLLECTION_INTERVAL_MINUTES} minutes")
    logger.info(f"Roads: {len(ROAD_LOCATIONS)} locations")
    logger.info(f"Output: {RAW_DATA_FILE}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    collection_count = 0
    total_success = 0

    try:
        while True:
            collection_count += 1
            logger.info(f"\n--- Collection #{collection_count} ---")

            success = collect_once()
            total_success += success

            # Stats
            total_possible = collection_count * 16
            success_rate = (total_success / total_possible) * 100 if total_possible > 0 else 0
            logger.info(f"Running stats: {total_success}/{total_possible} "
                       f"successful ({success_rate:.1f}%)")

            # Wait for next interval
            next_time = datetime.now(IST) + timedelta(minutes=COLLECTION_INTERVAL_MINUTES)
            logger.info(f"Next collection at: {next_time.strftime('%H:%M:%S IST')}")

            # Sleep until next collection
            time.sleep(COLLECTION_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        logger.info("\n=== Collection stopped by user ===")
        logger.info(f"Total collections: {collection_count}")
        logger.info(f"Total successful: {total_success}/{collection_count * 16}")
        logger.info(f"Data saved to: {RAW_DATA_FILE}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-Time Traffic Data Collector')
    parser.add_argument('--test', action='store_true', help='Test API connections')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously every 15 minutes')
    args = parser.parse_args()

    if args.test:
        test_apis()
    elif args.continuous:
        run_continuous()
    else:
        # Single collection run
        collect_once()
        print(f"\nData saved to: {RAW_DATA_FILE}")
        print("Run with --continuous for automated 15-minute collection.")
        print("Run with --test to check API connections.")
