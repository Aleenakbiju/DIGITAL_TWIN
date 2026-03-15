from flask import Flask, render_template, request, jsonify
import get_weather_data as dc
import pandas as pd
import requests
import traceback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Fetch Dynamic Weather Data ---
def get_weather_data(lat, lon, start_date, end_date):
    """Fetches historical weather from Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "temp": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind": data["hourly"]["wind_speed_10m"]
        })
        return df
    return None

# --- Multivariate AI Prediction ---
def predict_aqi_advanced(df_merged, model_name="random_forest", steps=24, forecast_start=None):
    # Drop rows where we don't even have past AQ data or weather features for training
    train = df_merged.dropna(subset=['val', 'temp', 'humidity', 'wind']).copy()
    
    if len(train) < 5: return []

    # Features: Air Quality, Temperature, Humidity, Wind Speed
    features = ['val', 'temp', 'humidity', 'wind']
    
    # Target is the NEXT hour's pollution
    train['target'] = train['val'].shift(-1)
    train_clean = train.dropna(subset=['target'])
    
    if len(train_clean) < 5: return []

    X = train_clean[features].values
    y = train_clean['target'].values
    
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
    model.fit(X, y)
    
    # Predict future using recursive forecasting
    predictions = []
    
    last_known_idx = train.index[-1]
    current_val = train.loc[last_known_idx, 'val']
    
    # Future data array
    if forecast_start is not None:
        future_data = df_merged[df_merged['datetime'] > forecast_start]
    else:
        future_data = df_merged.loc[last_known_idx + 1 :]
    
    # Fallback if no future weather
    if future_data.empty:
        last_row = train.loc[last_known_idx]
        for _ in range(steps):
             features_array = [[current_val, last_row['temp'], last_row['humidity'], last_row['wind']]]
             pred = model.predict(features_array)[0]
             predictions.append(round(pred, 2))
             current_val = pred
        return predictions

    for _, row in future_data.head(steps).iterrows():
        temp = row['temp'] if pd.notna(row['temp']) else 25.0
        humidity = row['humidity'] if pd.notna(row['humidity']) else 50.0
        wind = row['wind'] if pd.notna(row['wind']) else 5.0
        
        features_array = [[current_val, temp, humidity, wind]]
        pred = model.predict(features_array)[0]
        predictions.append(round(pred, 2))
        current_val = pred
        
    # If future_data has fewer rows than steps, pad the rest
    while len(predictions) < steps:
        features_array = [[current_val, 25.0, 50.0, 5.0]]
        pred = model.predict(features_array)[0]
        predictions.append(round(pred, 2))
        current_val = pred
        
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_locations')
def get_locations():
    query = request.args.get('q', 'Delhi')
    lat, lon = dc.get_coordinates(query)
    
    if lat is None: return jsonify([])

    url = f"{dc.BASE_URL}/locations"
    # Radius is 25000 to comply with OpenAQ API limits
    params = {"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 20}
    response = requests.get(url, headers=dc.headers, params=params)
    
    try:
        data = response.json()
        locations = data.get("results", []) if isinstance(data, dict) else []
        return jsonify(locations)
    except Exception as e:
        return jsonify([])

@app.route('/sync', methods=['POST'])
def sync_twin():
    try:
        data = request.json
        station_id = data.get('station_id')
        lat = data.get('lat')
        lon = data.get('lon')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        model_name = data.get('model', 'random_forest')
        
        aq_start = f"{start_date_str}T00:00:00Z"
        aq_end = f"{end_date_str}T23:59:59Z"

        # Fetch Weather (extend end date by 2 days to get future weather forecast)
        try:
            end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            extended_end_date_str = (end_date_dt + timedelta(days=2)).strftime("%Y-%m-%d")
        except:
            extended_end_date_str = end_date_str
            
        df_weather = get_weather_data(lat, lon, start_date_str, extended_end_date_str)

        sensors = dc.get_sensors(station_id)
        results = {}

        for sensor in sensors:
            name = sensor['parameter']['name']
            unit = sensor['parameter']['units']
            m_data = dc.get_measurements(sensor['id'], aq_start, aq_end)
            
            # We will process AQ data even if weather fails!
            if m_data:
                df_aq = pd.DataFrame([{
                    'datetime': pd.to_datetime(m['period']['datetimeTo']['utc']).tz_localize(None), 
                    'val': m['value']
                } for m in m_data])
                
                if not df_aq.empty:
                    df_aq['datetime'] = df_aq['datetime'].dt.round('H')
                    df_aq = df_aq.groupby('datetime')['val'].mean().reset_index()

                    # --- THE FIX IS HERE ---
                    if df_weather is not None and not df_weather.empty:
                        # 'outer' join ensures we keep future weather for prediction
                        df_merged = pd.merge(df_aq, df_weather, on='datetime', how='outer')
                        df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
                        # Fill any missing weather gaps
                        df_merged['temp'] = df_merged['temp'].ffill().bfill()
                        df_merged['humidity'] = df_merged['humidity'].ffill().bfill()
                        df_merged['wind'] = df_merged['wind'].ffill().bfill()
                    else:
                        # Fallback if weather completely fails
                        df_merged = df_aq.copy()
                        df_merged['temp'] = 25.0
                        df_merged['humidity'] = 50.0
                        df_merged['wind'] = 5.0
                    
                    if not df_merged.empty and len(df_merged) >= 5:
                        forecast_start_dt = pd.to_datetime(end_date_str + ' 23:59:59')
                        predicted_values = predict_aqi_advanced(df_merged, model_name=model_name, steps=24, forecast_start=forecast_start_dt)
                        
                        df_historical = df_merged.dropna(subset=['val'])
                        values = df_historical['val'].tolist()
                        timestamps = df_historical['datetime'].astype(str).tolist()

                        # Generate future timestamps anchored to user's selected end date
                        future_timestamps = [(forecast_start_dt + timedelta(hours=i+1)).dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(forecast_start_dt, 'dt') else (forecast_start_dt + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:%M:%S") for i in range(24)]

                        results[name] = {
                            "current": round(values[-1], 2) if values else 0,
                            "predicted_1hr": predicted_values[0] if predicted_values else "N/A",
                            "predicted_24hr": predicted_values,
                            "predicted_labels": future_timestamps,
                            "unit": unit,
                            "history": values, 
                            "labels": timestamps
                        }
                    else:
                        # Provide empty structural response so the frontend chart loop doesn't break
                        results[name] = {
                            "current": 0,
                            "predicted_1hr": "N/A",
                            "predicted_24hr": [],
                            "predicted_labels": [],
                            "unit": unit,
                            "history": [], 
                            "labels": []
                        }
                    
        return jsonify(results)
    except Exception as e:
        print(f"DEBUG CRASH: {traceback.format_exc()}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)