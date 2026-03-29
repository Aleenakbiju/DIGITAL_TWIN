import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import sys
import get_weather_data as dc

def fetch_and_save_real_data():
    print("\n" + "="*50)
    print("      OpenAQ REAL-TIME DATA ACQUISITION TOOL")
    print("="*50 + "\n")

    # 1. City Input & Geocoding
    city_query = input("📍 Step 1: Enter City Name (e.g., Delhi, London): ")
    print(f"Searching for coordinates for '{city_query}'...")
    lat, lon = dc.get_coordinates(city_query)
    
    if lat is None:
        print("❌ Error: Could not resolve coordinates for this city.")
        return

    print(f"✅ Found {city_query}: Lat {lat}, Lon {lon}")

    # 2. Station Discovery
    print(f"\n📍 Step 2: Fetching monitoring stations within 25km of {city_query}...")
    url = f"{dc.BASE_URL}/locations"
    params = {"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 10}
    response = requests.get(url, headers=dc.headers, params=params)
    
    try:
        data = response.json()
        stations = data.get("results", [])
    except:
        stations = []

    if not stations:
        print("❌ Error: No monitoring stations found in this area.")
        return

    print("\nAvailable Stations:")
    for idx, s in enumerate(stations):
        print(f"[{idx}] {s['name']} (ID: {s['id']})")
    
    try:
        choice = int(input(f"\nSelect a station index (0-{len(stations)-1}): "))
        selected_station = stations[choice]
    except:
        print("❌ Error: Invalid selection.")
        return

    station_id = selected_station['id']
    print(f"✅ Selected: {selected_station['name']}")

    # 3. Timeline Input
    print("\n📍 Step 3: Define Timeline (Format: YYYY-MM-DD)")
    start_date = input("Enter Start Date (e.g., 2024-01-01): ")
    end_date = input("Enter End Date (e.g., 2024-01-07): ")

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
        aq_start = f"{start_date}T00:00:00Z"
        aq_end = f"{end_date}T23:59:59Z"
    except:
        print("❌ Error: Invalid date format.")
        return

    # 4. Sensor Discovery
    print(f"\n📍 Step 4: Fetching available sensors for station {station_id}...")
    sensors = dc.get_sensors(station_id)
    if not sensors:
        print("❌ Error: No sensors available for this station.")
        return

    print(f"✅ Found {len(sensors)} sensors: {', '.join([s['parameter']['name'].upper() for s in sensors])}")

    # 5. Data Fetching with Progress Bar
    print("\n📍 Step 5: Acquiring Data From OpenAQ Cloud...")
    all_data = []
    
    total_sensors = len(sensors)
    for i, sensor in enumerate(sensors):
        sensor_id = sensor['id']
        param_name = sensor['parameter']['name']
        unit = sensor['parameter']['units']
        
        # Update Progress Bar for each sensor
        percent = int(((i + 1) / total_sensors) * 100)
        bar = "█" * (percent // 2) + "-" * (50 - percent // 2)
        sys.stdout.write(f"\r📡 Fetching {param_name.upper()}: |{bar}| {percent}%")
        sys.stdout.flush()
        
        m_data = dc.get_measurements(sensor_id, aq_start, aq_end)
        
        if m_data:
            for m in m_data:
                all_data.append({
                    "Timestamp": m['period']['datetimeTo']['utc'],
                    "Parameter": param_name,
                    "Value": m['value'],
                    "Unit": unit,
                    "Station": selected_station['name']
                })
        
        time.sleep(0.5) # Slight delay for visual progress effect

    if not all_data:
        print("\n❌ Error: No measurement data found for the selected range.")
        return

    # 6. Save to CSV
    print("\n\n📍 Step 6: Finalizing Result...")
    df = pd.DataFrame(all_data)
    
    # Pivot the data to match the wide format if needed, or keep it long
    # User wanted "CO, NO2, PM25" column style typically, so let's pivot
    try:
        df_pivot = df.pivot_table(index="Timestamp", columns="Parameter", values="Value").reset_index()
        filename = f"real_aqi_data_{city_query.replace(' ', '_')}.csv"
        df_pivot.to_csv(filename, index=False)
        print(f"✅ Success! Data saved to: {filename}")
        print(f"\nPreview of real-world data acquired:")
        print(df_pivot.head())
    except Exception as e:
        filename = f"real_aqi_data_{city_query.replace(' ', '_')}_raw.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Success! Data saved in raw format to: {filename}")
        print(df.head())

    print("\n" + "="*50)
    print("      DATA ACQUISITION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    fetch_and_save_real_data()
