import requests
from geopy.geocoders import Nominatim
import sys

# --- CONFIGURATION ---
API_KEY = "839c1de888b15bc2e82a27bbd80eb8b94ce52006100bc988d8adaee844b5865d"
BASE_URL = "https://api.openaq.org/v3"

headers = {
    "X-API-Key": API_KEY,
    "accept": "application/json"
}

def get_coordinates(place_name):
    """Converts a city name to Latitude/Longitude"""
    geolocator = Nominatim(user_agent="openaq_weather_fetcher")
    try:
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

def get_sensors(station_id):
    """Gets available sensors (pollutants) for a specific station"""
    url = f"{BASE_URL}/locations/{station_id}/sensors"
    response = requests.get(url, headers=headers)
    try:
        data = response.json()
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching sensors: {e}")
        return []

def get_measurements(sensors_id, start_date, end_date):
    """Fetches measurement data for a specific sensor"""
    url = f"{BASE_URL}/sensors/{sensors_id}/measurements"
    params = {
        "datetimeFrom": start_date,
        "datetimeTo": end_date,
        "limit": 1000 # Increased limit to get enough data for training
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10) # 10s timeout so dead sensors don't lock server
        response.raise_for_status()
        data = response.json()
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Failed to fetch sensor {sensors_id}: {e}")
        return []