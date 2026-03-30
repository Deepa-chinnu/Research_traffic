"""
=============================================================================
CONFIGURATION FILE FOR REAL-TIME TRAFFIC DATA COLLECTION
=============================================================================
IMPORTANT: Replace the placeholder API keys below with your actual keys.
           See API_SETUP_GUIDE.txt for how to get free API keys.
=============================================================================
"""

# ============================================================================
# API KEYS
# Reads from: 1) Environment variables (GitHub Actions / cloud)
#              2) Local .env file (for local development)
# Never hardcode keys here - use .env file instead.
# ============================================================================
import os as _os

def _load_env():
    """Load API keys from .env file if it exists."""
    env_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.env')
    if _os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    _os.environ.setdefault(key.strip(), val.strip())

_load_env()

TOMTOM_API_KEY = _os.environ.get("TOMTOM_API_KEY", "")
HERE_API_KEY = _os.environ.get("HERE_API_KEY", "")
OPENWEATHER_API_KEY = _os.environ.get("OPENWEATHER_API_KEY", "")

# ============================================================================
# COLLECTION SETTINGS
# ============================================================================
COLLECTION_INTERVAL_MINUTES = 15      # Collect every 15 minutes
DAILY_COLLECTIONS = 96                # 24 hours * 4 per hour = 96

# ============================================================================
# 16 ROAD LOCATIONS (Same roads as Kaggle dataset)
# Each entry: (Road Name, Area Name, Latitude, Longitude,
#              road_length_km, lanes, road_type)
# ============================================================================
ROAD_LOCATIONS = [
    # Indiranagar
    {
        "road": "100 Feet Road",
        "area": "Indiranagar",
        "lat": 12.9719,
        "lon": 77.6412,
        "road_length_km": 2.5,
        "lanes": 4,
        "road_type": "arterial",
        "max_capacity_vph": 4000,   # vehicles per hour (estimated)
    },
    {
        "road": "CMH Road",
        "area": "Indiranagar",
        "lat": 12.9686,
        "lon": 77.6403,
        "road_length_km": 1.8,
        "lanes": 4,
        "road_type": "arterial",
        "max_capacity_vph": 3500,
    },
    # M.G. Road
    {
        "road": "Trinity Circle",
        "area": "M.G. Road",
        "lat": 12.9716,
        "lon": 77.6197,
        "road_length_km": 0.5,
        "lanes": 6,
        "road_type": "junction",
        "max_capacity_vph": 5000,
    },
    {
        "road": "Anil Kumble Circle",
        "area": "M.G. Road",
        "lat": 12.9780,
        "lon": 77.5990,
        "road_length_km": 0.5,
        "lanes": 6,
        "road_type": "junction",
        "max_capacity_vph": 5000,
    },
    # Koramangala
    {
        "road": "Sony World Junction",
        "area": "Koramangala",
        "lat": 12.9340,
        "lon": 77.6165,
        "road_length_km": 0.5,
        "lanes": 4,
        "road_type": "junction",
        "max_capacity_vph": 4000,
    },
    {
        "road": "Sarjapur Road",
        "area": "Koramangala",
        "lat": 12.9100,
        "lon": 77.6680,
        "road_length_km": 3.0,
        "lanes": 4,
        "road_type": "arterial",
        "max_capacity_vph": 3800,
    },
    # Whitefield
    {
        "road": "Marathahalli Bridge",
        "area": "Whitefield",
        "lat": 12.9591,
        "lon": 77.7010,
        "road_length_km": 1.0,
        "lanes": 6,
        "road_type": "bridge",
        "max_capacity_vph": 5500,
    },
    {
        "road": "ITPL Main Road",
        "area": "Whitefield",
        "lat": 12.9857,
        "lon": 77.7263,
        "road_length_km": 2.0,
        "lanes": 4,
        "road_type": "arterial",
        "max_capacity_vph": 4000,
    },
    # Jayanagar
    {
        "road": "Jayanagar 4th Block",
        "area": "Jayanagar",
        "lat": 12.9250,
        "lon": 77.5830,
        "road_length_km": 1.5,
        "lanes": 4,
        "road_type": "arterial",
        "max_capacity_vph": 3500,
    },
    {
        "road": "South End Circle",
        "area": "Jayanagar",
        "lat": 12.9390,
        "lon": 77.5770,
        "road_length_km": 0.5,
        "lanes": 6,
        "road_type": "junction",
        "max_capacity_vph": 5000,
    },
    # Hebbal
    {
        "road": "Hebbal Flyover",
        "area": "Hebbal",
        "lat": 13.0358,
        "lon": 77.5970,
        "road_length_km": 2.0,
        "lanes": 6,
        "road_type": "flyover",
        "max_capacity_vph": 6000,
    },
    {
        "road": "Ballari Road",
        "area": "Hebbal",
        "lat": 13.0100,
        "lon": 77.5750,
        "road_length_km": 3.0,
        "lanes": 6,
        "road_type": "arterial",
        "max_capacity_vph": 5500,
    },
    # Yeshwanthpur
    {
        "road": "Tumkur Road",
        "area": "Yeshwanthpur",
        "lat": 13.0220,
        "lon": 77.5510,
        "road_length_km": 3.5,
        "lanes": 6,
        "road_type": "highway",
        "max_capacity_vph": 6000,
    },
    {
        "road": "Yeshwanthpur Circle",
        "area": "Yeshwanthpur",
        "lat": 13.0270,
        "lon": 77.5540,
        "road_length_km": 0.5,
        "lanes": 6,
        "road_type": "junction",
        "max_capacity_vph": 5000,
    },
    # Electronic City
    {
        "road": "Hosur Road",
        "area": "Electronic City",
        "lat": 12.8920,
        "lon": 77.6460,
        "road_length_km": 4.0,
        "lanes": 6,
        "road_type": "highway",
        "max_capacity_vph": 6500,
    },
    {
        "road": "Silk Board Junction",
        "area": "Electronic City",
        "lat": 12.9173,
        "lon": 77.6229,
        "road_length_km": 0.5,
        "lanes": 6,
        "road_type": "junction",
        "max_capacity_vph": 5500,
    },
]

# ============================================================================
# FILE PATHS
# ============================================================================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FILE = os.path.join(BASE_DIR, "data", "raw_traffic_data.csv")
DAILY_DATA_FILE = os.path.join(BASE_DIR, "data", "bangalore_realtime_daily.csv")
LOG_FILE = os.path.join(BASE_DIR, "logs", "collection.log")

# ============================================================================
# API ENDPOINTS
# ============================================================================
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TOMTOM_INCIDENTS_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"
HERE_FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Bangalore center for weather (one call covers the whole city)
BANGALORE_CENTER_LAT = 12.9716
BANGALORE_CENTER_LON = 77.5946
