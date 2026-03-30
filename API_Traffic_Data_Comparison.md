# Comprehensive Comparison: Real-Time Traffic & Weather Data APIs

## Research Context
- **Purpose**: Data collection for traffic research in Bangalore, India
- **Collection plan**: Querying every 15 minutes for 16 roads over 3 months
- **Total queries (3 months)**: 96 queries/day x 90 days x 16 roads = **138,240 API calls** (if 1 call per road per interval)
- **Date of research**: March 2026

---

## 1. GOOGLE MAPS PLATFORM APIs

### APIs Available
| API | Status | Purpose |
|-----|--------|---------|
| **Routes API (Compute Routes)** | Active (recommended) | Single origin-destination routing with traffic |
| **Routes API (Compute Route Matrix)** | Active (recommended) | Multi-origin/destination distance & time matrix |
| **Directions API** | Legacy (since March 2025) | Routing with traffic (being phased out) |
| **Distance Matrix API** | Legacy (since March 2025) | Distance/time matrix (being phased out) |
| **Roads API** | Active | GPS snapping, speed limits |
| **Roads Management Insights** | Enterprise only | Real-time streaming speed & congestion |

### Traffic Data Parameters Returned

| Parameter | Routes API | Directions API (Legacy) | Roads API |
|-----------|-----------|------------------------|-----------|
| **Travel time (no traffic)** | static_duration_in_seconds | duration | No |
| **Travel time (with traffic)** | duration_in_seconds | duration_in_traffic | No |
| **Speed categories** | NORMAL / SLOW / TRAFFIC_JAM (via speedReadingIntervals) | No | No |
| **Speed limit** | No | No | Yes (posted speed limit) |
| **Congestion level** | Via speed reading intervals on polyline | Implied by duration vs duration_in_traffic | No |
| **Traffic volume/flow count** | **No** | **No** | **No** |
| **Route polyline** | Yes (with traffic color-coding) | Yes | No |
| **Road closure** | Yes | Yes | No |

**Key capability**: The Routes API with `TRAFFIC_AWARE` or `TRAFFIC_AWARE_OPTIMAL` routing preference provides real-time and predictive traffic-based travel times. Speed reading intervals categorize segments as NORMAL, SLOW, or TRAFFIC_JAM.

### Pricing (Post March 2025)

| SKU | Price per 1,000 requests (USD) | Free monthly requests | Traffic data? |
|-----|-------------------------------|----------------------|---------------|
| Compute Routes - Essentials | $5.00 | 10,000 | No (basic only) |
| Compute Routes - Pro | $10.00 | 5,000 | Yes (TRAFFIC_AWARE) |
| Compute Routes - Enterprise | $15.00 | 1,000 | Yes + two-wheeler routing |
| Compute Route Matrix - Essentials | $5.00/element | 10,000 elements | No |
| Compute Route Matrix - Pro | $10.00/element | 5,000 elements | Yes |
| Compute Route Matrix - Enterprise | $15.00/element | 1,000 elements | Yes |

**India-specific pricing**: Up to 70% lower than global pricing. Indian developers can access up to $6,800 USD (~INR 5.7 Lakh) of free usage per month distributed across all products. Billing in INR is available.

### Rate Limits
- 3,000 QPM (queries per minute) for Compute Routes
- 3,000 EPM (elements per minute) for Compute Route Matrix
- Max 625 elements per matrix request (100 for TRAFFIC_AWARE_OPTIMAL)
- Max 25 intermediate waypoints per route

### 3-Month Cost Estimate (16 roads, every 15 min)

**Approach**: Use Compute Routes (Pro SKU) for each road segment.
- Total calls: 138,240 over 3 months (~46,080/month)
- Free: 5,000/month (Pro SKU)
- Billable: ~41,080/month
- Cost: 41,080 / 1,000 x $10 = **~$410.80/month = ~$1,232.40 for 3 months (USD)**
- **With India pricing (70% discount): ~$370 for 3 months**

**Alternative with Route Matrix**: If you batch 16 roads as origin-destination pairs per request, fewer API calls but same element count applies.

---

## 2. OLA MAPS API (India-Specific)

### APIs Available
- **Directions API** (with traffic) and **Directions Basic API** (without traffic)
- **Distance Matrix API** (with near real-time traffic) and **Distance Matrix Basic API** (without traffic)
- **Route Optimizer API**
- **Fleet Planner API**
- **Navigation SDK** (iOS/Android with turn-by-turn)

### Traffic Data Parameters Returned

| Parameter | Available? |
|-----------|-----------|
| **Travel time (with traffic)** | Yes (near real-time ETA) |
| **Distance** | Yes (routable distance) |
| **Speed** | Not explicitly documented as a separate field |
| **Congestion level** | Not explicitly documented |
| **Traffic volume/flow count** | **No** |
| **Route geometry** | Yes |
| **Alternative routes** | Yes (Directions API) |

**Note**: The Directions API provides routable paths with traffic data considerations and optimal paths. The Distance Matrix API provides near real-time ETA with traffic. The "Basic" variants of both APIs do NOT include real-time traffic.

### Pricing

| Tier | Details |
|------|---------|
| **Free tier** | Up to 500K API calls (some sources say 5 million/month for each API) |
| **Free for year** | Free access for 1 year for developers on Krutrim Cloud |
| **2-year free** | Up to 10 million calls for long-term (3+ year) commitments |
| **ONDC startups** | 3 years free for startups on India's ONDC platform |
| **Paid** | ~50% lower than Google Maps' India pricing |
| **Academic/NGO** | Discounts available (contact sales@olakrutrim.com) |

### Rate Limits
- Rate limiting enforced (429 response when exceeded)
- Specific limits not publicly documented; likely vary by plan

### India/Bangalore Availability
- **Built specifically for India** - designed for India's dynamic road conditions
- Ola itself uses Ola Maps for all its ride-hailing operations across India
- Excellent Bangalore coverage (Ola is headquartered in Bangalore)

### 3-Month Cost Estimate
- With 138,240 calls over 3 months: **Likely FREE** under the 500K or 5M free tier
- Academic discount may also apply
- **Estimated cost: $0 (free)**

---

## 3. TOMTOM TRAFFIC API

### APIs Available
- **Traffic Flow API** (Flow Segment Data, Raster/Vector Flow Tiles)
- **Traffic Incidents API** (Incident Details, Incident Tiles)
- **Traffic Stats API** (historical traffic analytics)

### Traffic Data Parameters Returned

| Parameter | Traffic Flow API | Traffic Incidents API |
|-----------|-----------------|----------------------|
| **Current speed** | Yes (`currentSpeed`) | No |
| **Free-flow speed** | Yes (`freeFlowSpeed`) | No |
| **Current travel time** | Yes (`currentTravelTime`) | No |
| **Free-flow travel time** | Yes (`freeFlowTravelTime`) | No |
| **Congestion level** | Implied (current vs free-flow speed ratio) | No |
| **Confidence score** | Yes (0 to 1) | No |
| **Road closure** | Yes (boolean) | Yes |
| **Functional road class** | Yes (`frc`) | No |
| **Road coordinates** | Yes (lat/lng array) | Yes (GeoJSON) |
| **Incident type** | No | Yes (accident, roadwork, etc.) |
| **Delay info** | No | Yes (delay length, significance) |
| **Traffic volume/flow count** | **No** | **No** |

**Update frequency**: Every 1 minute (flow), every 1 minute (incidents)

### Pricing

| Tier | Daily Free Allowance | Overage per 1,000 requests |
|------|---------------------|---------------------------|
| **Tile requests** (flow/incidents) | 50,000/day (shared) | EUR 0.08 |
| **Non-tile requests** (flow segment data, incident details) | 2,500/day (shared) | EUR 0.75 |

- No credit card required for Freemium plan
- Commercial use allowed on Freemium plan
- QPS limits: 5 to 50 depending on API

### India/Bangalore Availability
- **India is covered** by TomTom Traffic APIs
- **Bangalore is extensively covered** - TomTom Traffic Index ranks Bengaluru as the 2nd most congested city globally (congestion score 74.4%)
- Average speed data: 16.6 km/h in Bengaluru

### 3-Month Cost Estimate (16 roads, every 15 min)

**Using Flow Segment Data (non-tile endpoint)**:
- 96 queries/day x 16 roads = 1,536 non-tile requests/day
- Free allowance: 2,500/day -- **fits within free tier!**
- **Estimated cost: $0 (free) if staying within 2,500 non-tile requests/day**

**If exceeding free tier**:
- Overage at EUR 0.75/1,000 requests
- ~1,536 requests/day, all free if under 2,500/day

**BEST OPTION**: TomTom Flow Segment Data provides the richest traffic data (actual speed values, not just categories) and fits within the free tier for this use case.

---

## 4. HERE TRAFFIC API

### APIs Available
- **Traffic Flow API v7** (real-time speeds, congestion)
- **Traffic Incidents API v7** (accidents, road closures, construction)
- **Advanced Traffic** (deep coverage, lane-level flow)

### Traffic Data Parameters Returned

| Parameter | Traffic Flow API | Traffic Incidents API |
|-----------|-----------------|----------------------|
| **Speed** | Yes (`speed` in m/s) | No |
| **Speed (uncapped)** | Yes (`speedUncapped`, may exceed speed limit) | No |
| **Free-flow speed** | Yes (`freeFlow` speed) | No |
| **Jam Factor (congestion)** | Yes (0.0 = free flow, 10.0 = blocked) | No |
| **Jam Tendency** | Yes (increasing/decreasing/constant) | No |
| **Traversability** | Yes (open/closed/reversibleNotRoutable) | No |
| **Travel time** | Via speed and segment length calculation | No |
| **Road closure** | Via jamFactor = 10.0 | Yes |
| **Lane-level flow** | Yes (Advanced Traffic) | No |
| **Incident details** | No | Yes (type, severity, duration, location) |
| **Traffic volume/flow count** | **No** | **No** |

**Update frequency**: Every 1 minute (flow), every 2 minutes (incidents)

### Pricing

| Tier | Monthly Free Allowance | Overage |
|------|----------------------|---------|
| **Freemium** | 250,000 transactions/month | $1.00 per 1,000 transactions |
| **Advanced Traffic** | 2,500 transactions/month | $5.00 per 1,000 transactions |
| **Pro Plan** | 1,000,000 transactions/month | $449/month |

- No credit card required for Freemium
- ~5 QPS on Freemium plan
- No daily limit, monthly quota only

### Filtering Capabilities
- `minJamFactor` / `maxJamFactor` for congestion filtering
- `functionalClasses` for road type filtering
- Query by bounding box, circle with radius, or corridor

### India/Bangalore Availability
- **India is covered** - HERE has 96% map coverage in India
- 3,000+ person local team in India
- Extensive road coverage and real-time traffic data for Indian cities
- Smart City collaboration with Indian government
- EV charge-aware routing available in India

### 3-Month Cost Estimate (16 roads, every 15 min)

**Using Traffic Flow API (standard)**:
- Total calls: 138,240 over 3 months (46,080/month)
- Free allowance: 250,000/month
- **Estimated cost: $0 (free) -- fits entirely within free tier!**

**HERE is the most generous free tier** for this use case -- 250,000 free transactions/month easily covers 46,080 calls/month.

---

## 5. OPENSTREETMAP / OSRM

### Traffic Data Availability

| Feature | Available? |
|---------|-----------|
| **Real-time traffic** | **No** (not natively) |
| **Speed data** | Only posted speed limits from OSM tags |
| **Congestion** | **No** |
| **Travel time** | Yes (based on static speed profiles, not live traffic) |
| **Traffic volume/flow** | **No** |
| **Routing** | Yes (very fast, <1ms for Europe) |

### Key Points
- OSRM is a routing engine, not a traffic data source
- OSM data is completely **free** and open source
- OSRM supports importing external traffic data, but you must source it yourself
- No real-time traffic integration out of the box
- OpenTraffic (formerly a free global traffic dataset) is now defunct
- Datex2 provides free traffic data in the EU (not India)
- OSMTraffic project exists but is early-stage

### Cost: FREE (but no real-time traffic data)

### India/Bangalore
- OSM map data for Bangalore exists and is community-maintained
- No real-time traffic data available through OSM/OSRM for India

---

## 6. OTHER FREE/CHEAP APIs FOR INDIA/BANGALORE

### A. Mappls (MapmyIndia) -- India's Indigenous Mapping Platform

| Feature | Details |
|---------|---------|
| **Real-time traffic overlay** | Yes (color-coded: fast/medium/slow) |
| **Traffic-aware routing** | Yes (with real-time feeds and AI-driven ETA) |
| **Distance matrix** | Yes (with historic and live traffic) |
| **Speed data** | Average speed for regions |
| **Parameters returned** | Distance (meters), Duration (seconds), Geometry |
| **Bangalore-specific** | Live Traffic Signal Timing for 125+ smart signals |
| **Free tier** | Available (limited requests, exact count requires sign-up) |
| **Paid pricing** | ~$302-$605/month for 10K-50K calls (unconfirmed) |
| **Traffic volume/flow** | **No** |

**Unique for Bangalore**: Real-time traffic signal timing display (Green/Amber/Red countdown) at 125+ junctions -- India's first such feature.

### B. Open Data Sources (Free)

| Source | Data Available | Format |
|--------|---------------|--------|
| **OpenCity (data.opencity.in)** | Bengaluru traffic signal data, cycle ride data | CSV |
| **data.gov.in (OGD Platform)** | Government traffic datasets for India | Various |
| **Kaggle** | "Bangalore's Traffic Pulse" dataset | CSV |
| **Telraam** | Traffic count data (limited locations globally) | API |
| **GraphHopper Open Traffic Collection** | Curated list of open traffic data sources | Links |

### C. Mapbox Traffic Data
- Live speeds and typical speeds available
- Integrates with OSRM/Graphhopper routing engines
- **Enterprise plan only** (annual license per geographic region)
- Not practical for a free/cheap research project

---

## 7. OPENWEATHERMAP API

### APIs Available
- **One Call API 3.0** (current + forecast + historical + alerts)
- **Air Pollution API** (free, includes AQI and pollutant indices)
- **Geocoding API** (free)

### Weather Parameters Returned

| Category | Parameters |
|----------|-----------|
| **Current weather** | temp, feels_like, humidity, pressure, dew_point, clouds, visibility, wind_speed, wind_deg, uvi, weather description, sunrise, sunset |
| **Minutely forecast** | Precipitation (mm/h) for next 60 minutes |
| **Hourly forecast** | All current params + probability of precipitation (48 hours) |
| **Daily forecast** | Temp (min/max/day/night/eve/morn), feels_like, moon_phase, pop, weather description (8 days) |
| **Government alerts** | National weather warnings (source, description, severity) |
| **Air Pollution** | AQI, CO, NO, NO2, O3, SO2, NH3, PM2.5, PM10 |

### Pricing

| Plan | Free Allowance | Overage |
|------|---------------|---------|
| **One Call API 3.0** | 1,000 calls/day | EUR 0.14 per 100 calls |
| **Air Pollution API** | Free (included) | N/A |
| **Geocoding API** | Free (included) | N/A |

- Credit card required to sign up (set daily limit to 1,000 to avoid charges)
- Data updates every 10 minutes
- Historical data available (47+ years archive)

### 3-Month Cost Estimate
- For weather data: 96 calls/day (every 15 min) for 1 location (Bangalore)
- **Within 1,000 calls/day free tier -- $0 (free)**
- If querying 16 different precise locations: 96 x 16 = 1,536 calls/day
  - Exceeds free tier by 536 calls/day
  - Daily overage: 536/100 x EUR 0.14 = ~EUR 0.75/day
  - **3-month cost: ~EUR 68 (~$74 USD)**
  - **Alternative**: Query 1 location (Bangalore city center) since weather is similar across the city = **FREE**

---

## COMPREHENSIVE COMPARISON TABLE

### Data Parameters Comparison

| Parameter | Google Routes | Ola Maps | TomTom Flow | HERE Flow | OSM/OSRM | Mappls |
|-----------|:-------------|:---------|:-----------|:----------|:---------|:-------|
| **Current speed** | Categories only (NORMAL/SLOW/JAM) | No | Yes (km/h) | Yes (m/s) | No | Avg speed |
| **Free-flow speed** | No | No | Yes | Yes | No | No |
| **Travel time (live traffic)** | Yes | Yes (ETA) | Yes | Via calculation | No (static only) | Yes |
| **Travel time (no traffic)** | Yes | Yes (Basic API) | Yes (free-flow travel time) | Via free-flow speed | Yes | Yes |
| **Congestion level** | Categories (3 levels) | No explicit field | Ratio (current/freeflow) | Jam Factor (0-10) | No | Categories (fast/med/slow) |
| **Congestion trend** | No | No | No | Yes (increasing/decreasing/constant) | No | No |
| **Traffic volume/count** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** |
| **Road closure** | Yes | No | Yes | Yes | No | No |
| **Incidents** | No | No | Yes (separate API) | Yes (separate API) | No | No |
| **Confidence score** | No | No | Yes (0-1) | No | N/A | No |
| **Update frequency** | Real-time | Near real-time | 1 minute | 1-2 minutes | N/A | Few minutes |

**IMPORTANT NOTE**: None of the APIs provide actual traffic volume/vehicle flow counts. They provide speed and congestion estimates based on probe data (GPS traces from phones/vehicles), not physical vehicle counting.

### Pricing & Cost Comparison (3-Month Research Project)

| API | Free Tier | Est. 3-Month Cost | Best For |
|-----|-----------|-------------------|----------|
| **Google Routes (Pro)** | 5,000/month | ~$370 (India pricing) | Travel time with traffic |
| **Ola Maps** | 500K-5M calls | **$0 (FREE)** | India-specific routing + ETA |
| **TomTom Flow** | 2,500 non-tile/day | **$0 (FREE)** | Actual speed values + congestion |
| **HERE Flow** | 250,000/month | **$0 (FREE)** | Jam factor + speed + trend data |
| **OSM/OSRM** | Unlimited | **$0 (FREE)** | Static routing only (no traffic) |
| **Mappls** | Limited free | ~$0-$300 (unclear) | India-specific, signal timing |
| **OpenWeatherMap** | 1,000/day | **$0 (FREE)** if 1 location | Weather correlation data |

### India/Bangalore Availability

| API | India Coverage | Bangalore Specific |
|-----|---------------|-------------------|
| **Google Routes** | Yes (India-specific pricing) | Yes |
| **Ola Maps** | Built for India | Excellent (HQ in Bangalore) |
| **TomTom** | Major cities covered | Yes (Traffic Index data exists) |
| **HERE** | 96% map coverage, 3,000 local staff | Yes |
| **OSM/OSRM** | Community-maintained maps | Yes (maps only, no traffic) |
| **Mappls** | 66 lakh km of roads, 7,000 cities | Yes + Live Signal Timing |
| **OpenWeatherMap** | Global | Yes |

---

## RECOMMENDATIONS FOR RESEARCH PROJECT

### Best Strategy: Multi-API Approach (All Free)

For a 3-month research project collecting traffic data every 15 minutes for 16 roads in Bangalore:

#### Primary Traffic Data Collection (FREE)
1. **TomTom Traffic Flow API** -- Use Flow Segment Data endpoint
   - Returns: currentSpeed, freeFlowSpeed, currentTravelTime, freeFlowTravelTime, confidence
   - 1,536 calls/day fits within 2,500 free daily non-tile allowance
   - Richest speed data available for free
   - **Cost: $0**

2. **HERE Traffic Flow API** -- Use as secondary/validation source
   - Returns: speed, freeFlow speed, jamFactor (0-10), jamTendency
   - 46,080 calls/month fits within 250,000 free monthly allowance
   - Best congestion metric (jamFactor is a standardized 0-10 scale)
   - **Cost: $0**

3. **Ola Maps Distance Matrix API** -- For India-specific ETA data
   - Returns: distance, duration with near real-time traffic
   - Well within generous free tier
   - Tailored for Indian road conditions
   - **Cost: $0**

#### Supplementary Data (FREE)
4. **OpenWeatherMap One Call API 3.0** -- For weather correlation
   - Query 1 Bangalore location every 15 min (96 calls/day)
   - Returns: temp, humidity, precipitation, wind, visibility
   - Useful for analyzing weather-traffic correlation
   - **Cost: $0**

5. **TomTom Traffic Incidents API** -- For incident data
   - Returns: accident/roadwork/closure details
   - Shares the 2,500 daily non-tile allowance with Flow API
   - Budget: allocate ~964 calls to incidents (remaining after 1,536 flow calls)
   - **Cost: $0**

#### Total Estimated Cost: $0 (all within free tiers)

### What You CANNOT Get From Any API
- **Actual vehicle counts/traffic volume** -- No API provides this. You would need physical sensors (loop detectors, cameras with CV, pneumatic tubes) or commercial datasets (xMap, Datacorp).
- **Vehicle classification counts** -- Not available via APIs
- **Lane-level occupancy** -- Not available (except HERE Advanced Traffic at $5/1K, limited)

### If Budget Is Available (~$400)
- Add **Google Routes API (Pro)** for comparison with Google's traffic-aware travel times
- Use India-specific pricing for lower cost

---

## API ENDPOINT QUICK REFERENCE

### TomTom Flow Segment Data
```
GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=12.9716,77.5946&key=YOUR_KEY
```
Response: currentSpeed, freeFlowSpeed, currentTravelTime, freeFlowTravelTime, confidence, coordinates

### HERE Traffic Flow
```
GET https://data.traffic.hereapi.com/v7/flow?in=circle:12.9716,77.5946;r=500&apiKey=YOUR_KEY
```
Response: speed, speedUncapped, freeFlow, jamFactor, jamTendency, traversability

### Ola Maps Distance Matrix
```
GET https://api.olamaps.io/routing/v1/distanceMatrix?origins=lat,lng&destinations=lat,lng&api_key=YOUR_KEY
```
Response: distance, duration (with traffic)

### OpenWeatherMap One Call 3.0
```
GET https://api.openweathermap.org/data/3.0/onecall?lat=12.9716&lon=77.5946&appid=YOUR_KEY
```
Response: temp, humidity, pressure, wind_speed, clouds, visibility, weather, precipitation
