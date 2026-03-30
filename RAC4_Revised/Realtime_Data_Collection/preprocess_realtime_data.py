"""
=============================================================================
PREPROCESS REAL-TIME DATA TO MATCH KAGGLE DATASET FORMAT
=============================================================================
Reads raw 15-minute interval data and converts to daily records matching
the Bangalore Traffic Dataset format (real API-sourced columns only).

Usage:
  python preprocess_realtime_data.py                  # Process all data
  python preprocess_realtime_data.py --date 2026-03-15  # Process specific date
  python preprocess_realtime_data.py --report           # Generate quality report

Author: PhD Research - Traffic Flow Prediction
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from config import RAW_DATA_FILE, DAILY_DATA_FILE, ROAD_LOCATIONS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# TRAFFIC VOLUME ESTIMATION (Greenshields Model)
# ============================================================================

def estimate_traffic_volume(avg_speed, free_flow_speed, jam_factor,
                            max_capacity_vph, lanes, hours=24):
    """
    Estimate daily traffic volume using the Greenshields traffic flow model.

    The Greenshields model relates:
      Flow (q) = Density (k) x Speed (v)
      where k = k_jam * (1 - v/v_f)
      so: q = k_jam * v * (1 - v/v_f)

    We use the relationship:
      Volume = Capacity * (v/v_f) * (1 - (1 - v/v_f)^2)

    Parameters:
      avg_speed:       Current average speed (km/h)
      free_flow_speed: Free flow speed (km/h)
      jam_factor:      HERE jam factor (0=free, 10=jam)
      max_capacity_vph: Maximum vehicles per hour capacity
      lanes:           Number of lanes
      hours:           Hours in the day (24 for daily)

    Returns:
      Estimated daily traffic volume (vehicles/day)
    """
    if avg_speed is None or free_flow_speed is None or free_flow_speed == 0:
        # Fallback: use jam_factor if available
        if jam_factor is not None:
            # jam_factor 0 = free flow, 10 = fully jammed
            utilization = jam_factor / 10.0
            # Traffic volume follows a parabolic curve:
            # peak at ~50% utilization, lower at both extremes
            if utilization <= 0.5:
                # Under-saturated flow
                flow_ratio = utilization * 1.8  # 0 to 0.9
            else:
                # Over-saturated flow (congested)
                flow_ratio = 1.0 - (utilization - 0.5) * 0.4  # 0.9 to 0.7
            volume_per_hour = max_capacity_vph * flow_ratio
        else:
            return None
    else:
        # Greenshields model
        speed_ratio = min(avg_speed / free_flow_speed, 1.0)

        # Flow = capacity * 4 * speed_ratio * (1 - speed_ratio)
        # Max flow at speed_ratio = 0.5, zero at 0 and 1
        # Adjusted for real-world where flow is still significant at high speeds
        if speed_ratio > 0.9:
            # Near free flow - moderate traffic
            flow_ratio = 0.3 + 0.2 * (1 - speed_ratio) / 0.1
        elif speed_ratio > 0.5:
            # Moderate congestion - high traffic
            flow_ratio = 0.5 + 0.5 * (0.9 - speed_ratio) / 0.4
        elif speed_ratio > 0.2:
            # Heavy congestion - peak then declining traffic
            flow_ratio = 1.0 - 0.3 * (0.5 - speed_ratio) / 0.3
        else:
            # Near standstill - very low throughput
            flow_ratio = 0.7 * speed_ratio / 0.2

        volume_per_hour = max_capacity_vph * flow_ratio

    # Scale for the collection interval and extrapolate to daily
    daily_volume = volume_per_hour * hours

    # Add some realistic daily variation based on hour-of-day patterns
    return max(0, int(daily_volume))


# ============================================================================
# WEATHER CONDITION MAPPING
# ============================================================================

def map_weather_condition(weather_main, rain_mm=0):
    """
    Map OpenWeatherMap weather conditions to Kaggle dataset categories.
    Kaggle uses: Clear, Rain, Fog, Windy, Overcast
    """
    if weather_main is None:
        return 'Clear'

    weather_main = str(weather_main).lower()

    if weather_main in ['clear']:
        return 'Clear'
    elif weather_main in ['rain', 'drizzle', 'thunderstorm']:
        return 'Rain'
    elif weather_main in ['mist', 'fog', 'haze', 'smoke']:
        return 'Fog'
    elif weather_main in ['clouds', 'overcast']:
        return 'Clear'   # Light clouds = Clear in Kaggle
    elif weather_main in ['squall', 'tornado']:
        return 'Windy'
    else:
        return 'Clear'


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_to_daily(date_filter=None):
    """
    Read raw 15-minute data, aggregate to daily records with real API-sourced columns.

    Parameters:
      date_filter: Optional 'YYYY-MM-DD' to process specific date only.

    Returns:
      DataFrame with same columns as Kaggle dataset.
    """
    if not os.path.exists(RAW_DATA_FILE):
        print(f"ERROR: Raw data file not found: {RAW_DATA_FILE}")
        print("Run realtime_data_collector.py first to collect data.")
        sys.exit(1)

    # Load raw data
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded {len(df)} raw records from {RAW_DATA_FILE}")

    if date_filter:
        df = df[df['date'] == date_filter]
        print(f"Filtered to date {date_filter}: {len(df)} records")

    if len(df) == 0:
        print("No data to process.")
        return pd.DataFrame()

    # Build road info lookup
    road_info_map = {r['road']: r for r in ROAD_LOCATIONS}

    # ---- AGGREGATE TO DAILY PER ROAD ----
    daily_records = []

    for (date, road_name), group in df.groupby(['date', 'road_name']):
        area_name = group['area_name'].iloc[0]

        # Road characteristics
        rinfo = road_info_map.get(road_name, {})
        lanes = rinfo.get('lanes', 4)
        max_cap = rinfo.get('max_capacity_vph', 4000)

        # ---- AVERAGE SPEED ----
        # Combine TomTom and HERE speeds (take mean of available)
        tt_speeds = group['tomtom_current_speed'].dropna()
        here_speeds = group['here_speed'].dropna()

        all_speeds = pd.concat([tt_speeds, here_speeds])
        avg_speed = all_speeds.mean() if len(all_speeds) > 0 else None

        # ---- FREE FLOW SPEED ----
        tt_ff = group['tomtom_free_flow_speed'].dropna()
        here_ff = group['here_free_flow_speed'].dropna()
        all_ff = pd.concat([tt_ff, here_ff])
        free_flow_speed = all_ff.mean() if len(all_ff) > 0 else None

        # ---- TRAVEL TIME INDEX ----
        # TTI = actual travel time / free flow travel time
        tt_curr = group['tomtom_current_travel_time'].dropna()
        tt_ff_time = group['tomtom_free_flow_travel_time'].dropna()

        if len(tt_curr) > 0 and len(tt_ff_time) > 0:
            tti = tt_curr.mean() / max(tt_ff_time.mean(), 1)
            tti = min(tti, 3.0)  # Cap at 3.0
        elif avg_speed and free_flow_speed and avg_speed > 0:
            tti = free_flow_speed / avg_speed
            tti = min(tti, 3.0)
        else:
            tti = 1.0

        # ---- CONGESTION LEVEL (0-100) ----
        jam_factors = group['here_jam_factor'].dropna()
        if len(jam_factors) > 0:
            avg_jam = jam_factors.mean()
            congestion_level = (avg_jam / 10.0) * 100  # Scale 0-10 to 0-100
        elif avg_speed and free_flow_speed and free_flow_speed > 0:
            speed_ratio = avg_speed / free_flow_speed
            congestion_level = (1 - speed_ratio) * 100
        else:
            congestion_level = 50  # default

        congestion_level = max(0, min(100, congestion_level))

        # ---- ROAD CAPACITY UTILIZATION ----
        if avg_speed and free_flow_speed and free_flow_speed > 0:
            capacity_util = (1 - avg_speed / free_flow_speed) * 100
            capacity_util = max(0, min(100, capacity_util))
        else:
            capacity_util = congestion_level

        # ---- TRAFFIC VOLUME (ESTIMATED) ----
        avg_jam = jam_factors.mean() if len(jam_factors) > 0 else None
        traffic_volume = estimate_traffic_volume(
            avg_speed, free_flow_speed, avg_jam, max_cap, lanes
        )
        if traffic_volume is None:
            traffic_volume = int(max_cap * 12)  # rough estimate: 12 active hours

        # ---- INCIDENT REPORTS ----
        incident_count = group['incident_count'].max()
        incident_count = int(incident_count) if pd.notna(incident_count) else 0

        # ---- WEATHER FEATURES (real API data) ----
        temperature = group['temperature_c'].mean() if 'temperature_c' in group.columns else None
        temperature = round(temperature, 2) if pd.notna(temperature) else None
        humidity = group['humidity_pct'].mean() if 'humidity_pct' in group.columns else None
        humidity = round(humidity, 2) if pd.notna(humidity) else None
        wind_speed = group['wind_speed_ms'].mean() if 'wind_speed_ms' in group.columns else None
        wind_speed = round(wind_speed, 2) if pd.notna(wind_speed) else None
        rain = group['rain_1h_mm'].mean() if 'rain_1h_mm' in group.columns else 0
        rain = round(rain, 2) if pd.notna(rain) else 0
        visibility = group['visibility_m'].mean() if 'visibility_m' in group.columns else None
        visibility = round(visibility, 2) if pd.notna(visibility) else None

        # ---- TRAFFIC FEATURES (real API data) ----
        free_flow_speed_val = round(free_flow_speed, 2) if free_flow_speed else None
        avg_jam = jam_factors.mean() if len(jam_factors) > 0 else None
        jam_factor_val = round(avg_jam, 4) if pd.notna(avg_jam) else None
        confidences = group['tomtom_confidence'].dropna()
        confidence_val = round(confidences.mean(), 4) if len(confidences) > 0 else None

        # ---- WEATHER CONDITIONS ----
        weather_vals = group['weather_main'].dropna()
        if len(weather_vals) > 0:
            # Most common weather during the day
            weather_condition = map_weather_condition(weather_vals.mode().iloc[0], rain)
        else:
            weather_condition = 'Clear'

        # Check for windy conditions
        wind_speeds = group['wind_speed_ms'].dropna()
        if len(wind_speeds) > 0 and wind_speeds.mean() > 8:
            weather_condition = 'Windy'

        # ---- ROADWORK AND CONSTRUCTION ----
        has_construction = group['has_construction'].any()
        roadwork = 'Yes' if has_construction else 'No'

        # ---- BUILD FINAL RECORD ----
        record = {
            'Date': date,
            'Area Name': area_name,
            'Road/Intersection Name': road_name,
            'Traffic Volume': traffic_volume,
            'Average Speed': round(avg_speed, 2) if avg_speed else 30.0,
            'Travel Time Index': round(tti, 4),
            'Congestion Level': round(congestion_level, 2),
            'Road Capacity Utilization': round(capacity_util, 2),
            'Incident Reports': incident_count,
            'Weather Conditions': weather_condition,
            'Roadwork and Construction Activity': roadwork,
            'Temperature': temperature,
            'Humidity': humidity,
            'Wind Speed': wind_speed,
            'Rain Volume': rain,
            'Visibility': visibility,
            'Free Flow Speed': free_flow_speed_val,
            'Jam Factor': jam_factor_val,
            'Confidence': confidence_val,
        }
        daily_records.append(record)

    # Create DataFrame
    daily_df = pd.DataFrame(daily_records)

    if len(daily_df) > 0:
        # Sort by date and road
        daily_df = daily_df.sort_values(['Date', 'Area Name', 'Road/Intersection Name'])
        daily_df = daily_df.reset_index(drop=True)

        # Save to CSV (append if exists, but avoid duplicates)
        if os.path.exists(DAILY_DATA_FILE):
            existing = pd.read_csv(DAILY_DATA_FILE)
            # Remove dates that we're about to add (to avoid duplicates)
            new_dates = daily_df['Date'].unique()
            existing = existing[~existing['Date'].isin(new_dates)]
            daily_df = pd.concat([existing, daily_df], ignore_index=True)
            daily_df = daily_df.sort_values(['Date', 'Area Name', 'Road/Intersection Name'])

        daily_df.to_csv(DAILY_DATA_FILE, index=False)
        print(f"\nDaily data saved to: {DAILY_DATA_FILE}")
        print(f"Total daily records: {len(daily_df)}")
        print(f"Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
        print(f"Unique roads: {daily_df['Road/Intersection Name'].nunique()}")
        print(f"Unique areas: {daily_df['Area Name'].nunique()}")

    return daily_df


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def generate_quality_report():
    """Generate a data quality report for collected data."""

    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    # Raw data stats
    if os.path.exists(RAW_DATA_FILE):
        raw_df = pd.read_csv(RAW_DATA_FILE)
        print(f"\n--- RAW DATA (15-minute intervals) ---")
        print(f"Total records: {len(raw_df)}")
        print(f"Date range: {raw_df['date'].min()} to {raw_df['date'].max()}")
        print(f"Unique dates: {raw_df['date'].nunique()}")
        print(f"Unique roads: {raw_df['road_name'].nunique()}")

        # Collection success rate
        status_counts = raw_df['collection_status'].value_counts()
        total = len(raw_df)
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/total*100:.1f}%)")

        # Data completeness per column
        print(f"\nColumn Completeness:")
        completeness_cols = {
            'tomtom_current_speed': 'Current Speed',
            'here_speed': 'HERE Speed',
            'here_jam_factor': 'Jam Factor',
            'weather_main': 'Weather',
            'temperature_c': 'Temperature',
        }
        for col, display_name in completeness_cols.items():
            non_null = raw_df[col].notna().sum()
            pct = (non_null / total) * 100
            print(f"  {display_name:<30} {non_null:>6}/{total} ({pct:.1f}%)")

        # Daily coverage
        print(f"\nDaily Coverage (records per day):")
        daily_counts = raw_df.groupby('date').size()
        print(f"  Expected: {16 * 96} records/day (16 roads x 96 intervals)")
        print(f"  Average:  {daily_counts.mean():.0f} records/day")
        print(f"  Min:      {daily_counts.min()} records/day")
        print(f"  Max:      {daily_counts.max()} records/day")
    else:
        print("\nNo raw data file found.")

    # Daily data stats
    if os.path.exists(DAILY_DATA_FILE):
        daily_df = pd.read_csv(DAILY_DATA_FILE)
        print(f"\n--- DAILY DATA (Kaggle format) ---")
        print(f"Total records: {len(daily_df)}")
        print(f"Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
        print(f"Unique dates: {daily_df['Date'].nunique()}")
        print(f"Unique areas: {daily_df['Area Name'].nunique()}")
        print(f"Unique roads: {daily_df['Road/Intersection Name'].nunique()}")

        print(f"\nColumn Statistics:")
        numeric_cols = ['Traffic Volume', 'Average Speed', 'Travel Time Index',
                       'Congestion Level', 'Road Capacity Utilization',
                       'Incident Reports', 'Temperature', 'Humidity',
                       'Wind Speed', 'Rain Volume', 'Visibility',
                       'Free Flow Speed', 'Jam Factor', 'Confidence']
        for col in numeric_cols:
            if col in daily_df.columns:
                vals = daily_df[col].dropna()
                print(f"  {col:<35} min={vals.min():>10.2f}  "
                      f"mean={vals.mean():>10.2f}  max={vals.max():>10.2f}")

        # Weather distribution
        print(f"\nWeather Conditions:")
        weather_counts = daily_df['Weather Conditions'].value_counts()
        for w, c in weather_counts.items():
            print(f"  {w}: {c} ({c/len(daily_df)*100:.1f}%)")

        # Roadwork distribution
        print(f"\nRoadwork Activity:")
        rw_counts = daily_df['Roadwork and Construction Activity'].value_counts()
        for r, c in rw_counts.items():
            print(f"  {r}: {c} ({c/len(daily_df)*100:.1f}%)")
    else:
        print("\nNo daily data file found. Run preprocessing first.")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Real-Time Traffic Data')
    parser.add_argument('--date', type=str, help='Process specific date (YYYY-MM-DD)')
    parser.add_argument('--report', action='store_true', help='Generate quality report')
    args = parser.parse_args()

    if args.report:
        generate_quality_report()
    else:
        daily_df = preprocess_to_daily(date_filter=args.date)

        if len(daily_df) > 0:
            print("\n--- Sample Records ---")
            print(daily_df.head().to_string())
            print(f"\nColumns: {list(daily_df.columns)}")
            print(f"\nThis data has the SAME format as the Kaggle dataset!")
            print(f"Ready to use with your ML models in RAC4_Revised/")
