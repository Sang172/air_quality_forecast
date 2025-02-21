import requests
from datetime import datetime, timedelta, timezone
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from geopy.distance import geodesic
import time
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from dotenv import load_dotenv
import os
import logging
import s3fs
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import tempfile

load_dotenv()

API_KEY=os.environ.get('OPENWEATHER_API_KEY')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

logger.info('initializing s3fs')
s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

with s3.open('s3://air-quality-forecast/scaler_feature.pickle', 'rb') as file:
    scaler_feature = pickle.load(file)
logger.info("Loaded feature scaler from S3")

with s3.open('s3://air-quality-forecast/scaler_target.pickle', 'rb') as file:
    scaler_target = pickle.load(file)
logger.info("Loaded target scaler from S3")


def download_from_s3_s3fs(s3_file_name, local_path):
    try:
        s3_path = f"s3://{S3_BUCKET_NAME}/{s3_file_name}"
        with s3.open(s3_path, 'rb') as s3_file, open(local_path, 'wb') as local_file:
            local_file.write(s3_file.read())
        logger.info(f"Downloaded {s3_file_name} from S3 to {local_path}")
    except Exception as e:
        logger.exception(f"Error downloading {s3_file_name} from S3: {e}")
        raise

MODEL_FILE_NAME = 'lstm.keras'
LOCAL_MODEL_PATH = os.path.join(tempfile.gettempdir(), 'local_lstm.keras')
download_from_s3_s3fs(MODEL_FILE_NAME, LOCAL_MODEL_PATH)
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
logger.info('all data loaded from S3 Bucket')


def geocode_address(address):
    geolocator = Nominatim(user_agent="openaq_data_fetcher")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
        logger.info(f"Geocoding error: {e}")
        return None
    


def unixtime_to_local(dt):
    utc_dt = datetime.fromtimestamp(dt, tz=timezone.utc)

    pacific = pytz.timezone("America/Los_Angeles")
    local_dt = utc_dt.astimezone(pacific)

    return local_dt.isoformat()




def open_weather_api_call(coord, retries=5, delay=0.5):
    latitude = coord[0]
    longitude = coord[1]
    c = (np.round(latitude,6),np.round(longitude,6))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={int(time.time())-172800}&end={int(time.time())}&appid={API_KEY}"

    for attempt in range(retries):
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()['list']
            lst = []
            for x in data:
                for k,v in x.items():
                    if k=='dt':
                        a = unixtime_to_local(v)
                    if k=='components':
                        b = v['pm2_5']
                lst.append((a,b,c))
            lst = sorted(lst, key=lambda x: x[0])
            return lst
        else:
            print(f"Attempt {attempt + 1} failed: {response.status_code} - {response.text}")
            if attempt < retries - 1:
                time.sleep(delay)

    print("All retry attempts failed.")
    return None 
        




def create_dataframe(data_list):

    processed_data = []
    for ts, val, coord in data_list:
        dt = datetime.fromisoformat(ts)
        processed_data.append((dt, val, coord))

    latest_timestamp = max(processed_data, key=lambda item: item[0])[0]

    desired_timestamps = [latest_timestamp - timedelta(hours=i) for i in range(47, -1, -1)]

    desired_time_format = "%Y-%m-%d %H:%M:%S"
    formatted_timestamps = [dt.strftime(desired_time_format) for dt in desired_timestamps]

    df_data = {
        'aq_coord': [processed_data[0][2]] * 48,
        'time': formatted_timestamps,
        'value': [None] * 48,
        'aq_lat': [processed_data[0][2][0]] * 48,
        'aq_lon': [processed_data[0][2][1]] * 48
    }
    df = pd.DataFrame(df_data)
    df['time'] = pd.to_datetime(df['time'])

    data_dict = {timestamp.replace(tzinfo=None): value for timestamp, value, _ in processed_data}
    for i in range(len(df)):
        if df['time'][i] in data_dict:
            df.loc[i, 'value'] = data_dict[df['time'][i]]

    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].interpolate(method='linear', limit_direction='both')

    return df.tail(24).reset_index(drop=True)




def open_meteo_api_call(measurements, hours_ago=48):

    now_utc = datetime.now(pytz.utc)
    pacific_tz = pytz.timezone('US/Pacific')
    now_pacific = now_utc.astimezone(pacific_tz)
    date_to = now_pacific.isoformat()
    date_from = (now_pacific - timedelta(hours=hours_ago//2)).isoformat()

    coord = measurements[0][2]
    aq_lat = coord[0]
    aq_lon = coord[1]

    url1 = "https://api.open-meteo.com/v1/forecast?"
    url2 = f"latitude={aq_lat}&longitude={aq_lon}"
    url3 = f"&start_date={date_from[:10]}&end_date={date_to[:10]}&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=America%2FLos_Angeles"
    url = url1+url2+url3

    df = create_dataframe(measurements)

    max_retries = 5
    delay = 0.5

    for attempt in range(max_retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                df1 = pd.DataFrame(data['hourly'])
                df1['w_lat'] = data['latitude']
                df1['w_lon'] = data['longitude']
                df1['w_elevation'] = data['elevation']
                df1['w_coord'] = df1.apply(lambda x: (x['w_lat'], x['w_lon']), axis=1)
                df1['time'] = pd.to_datetime(df1['time'])
                df1['distance_mi'] = df1.apply(lambda x: geodesic(x['w_coord'], coord).miles, axis=1)
                df = df.merge(df1, left_on=['time'], right_on=['time'], how='left')
                df['lat_diff'] = df['aq_lat'] - df['w_lat']
                df['lon_diff'] = df['aq_lon'] - df['w_lon']
                df = df.drop(columns=['w_lat', 'w_lon', 'w_coord'])
                return df
            else:
                print(f"Attempt {attempt + 1}/{max_retries} - Error: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Request Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            continue
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Unexpected Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            continue


    print(f"Failed to retrieve data after {max_retries} attempts.")
    return None




def cyclical_encoding(df1):
    df = df1.copy()
    
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek
    df['dayofyear'] = df['time'].dt.dayofyear
    df['hour'] = df['time'].dt.hour

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['wind_direction_10m_sin'] = np.sin(2 * np.pi * df['wind_direction_10m'] / 360)
    df['wind_direction_10m_cos'] = np.cos(2 * np.pi * df['wind_direction_10m'] / 360)

    df = df.drop(columns = ['month','dayofweek','dayofyear','hour','wind_direction_10m'])
    return df




def normalize_data(df, feature_num, target):
    df[feature_num] = scaler_feature.transform(df[feature_num])
    df[feature_num] = df[feature_num].interpolate(method='linear').bfill().ffill()
    df[target] = scaler_target.transform(df[target])
    df[target] = df[target].interpolate(method='linear').bfill().ffill()
    return df


def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy"),
        (150.5, 250.4, 201, 300, "Very Unhealthy"),
        (250.5, 350.4, 301, 400, "Hazardous"),
        (350.5, 500.4, 401, 500, "Hazardous"),
        (500.5, float('inf'), 501, 501, "Hazardous")
    ]

    for bp_low, bp_high, aqi_low, aqi_high, interpretation in breakpoints:
        if bp_low <= pm25 <= bp_high:
            aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
            return round(aqi), interpretation
        



def generate_next_24_hours(datetime_string):
    start_datetime = datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')
    hourly_datetimes = []
    for i in range(24):
        next_hour = start_datetime + timedelta(hours=i+1)
        hourly_datetimes.append(next_hour.strftime('%Y-%m-%d %H:%M:%S'))
    return hourly_datetimes




def predict(data, target, feature_num):
    X = data[target+feature_num].values
    X = np.expand_dims(X, axis=0)
    mean = scaler_target.mean_
    std = scaler_target.scale_
    pred = model.predict(X)[0] * std + mean
    latest = data.iloc[-1]['time']
    forecasts = generate_next_24_hours(str(latest))
    forecasts = {forecasts[i]: pm25_to_aqi(pred[i]) for i in range(24)}
    return forecasts



def main(address: str):
    logger.info(f"Starting main process for address: {address}")

    logger.info("Geocoding address...")
    coord = geocode_address(address)
    if not coord:
        logger.error("Could not geocode the provided address.")
        return None
    logger.info(f"Geocoord is {coord}")

    logger.info("Getting closest air quality data...")
    measurements = open_weather_api_call(coord)
    if not measurements:
        logger.error("Could not find close air quality measurements")
        return None
    logger.info("Retrieved air quality data.")

    logger.info("Getting weather data...")
    data = open_meteo_api_call(measurements)
    if data is None:
        logger.error("Could not retrieve weather data.")
        return None
    logger.info("Retrieved weather data.")


    logger.info("Start feature engineering")
    id_columns = ['aq_coord', 'time']
    target = ['value']
    feature_num = ['aq_lat', 'aq_lon', 'lat_diff', 'lon_diff', 'distance_mi', 'temperature_2m', 'relative_humidity_2m',
                   'precipitation', 'rain', 'snowfall', 'cloud_cover', 'w_elevation', 'wind_speed_10m',
                   'wind_direction_10m']
    data = data[id_columns + target + feature_num]
    data = cyclical_encoding(data)
    feature_num.pop(-1)
    feature_num += ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                    'dayofyear_sin', 'dayofyear_cos', 'wind_direction_10m_cos', 'wind_direction_10m_sin']
    data = normalize_data(data, feature_num, target)
    logger.info("Feature engineering complete.")

    logger.info("Getting prediction...")
    forecasts = predict(data, target, feature_num)
    if not forecasts:
        logger.error("Could not generate the forecast.")
        return None
    logger.info("Prediction complete.")
    logger.info(f"Returning forecasts")
    return forecasts



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        address = request.form['address']
        forecast_data = main(address)
        return render_template('index.html', forecast=forecast_data, address=address)
    return render_template('index.html', forecast=None, address="")

@app.route('/forecast', methods=['POST'])
def forecast():
    address = request.form['address']
    forecast_data = main(address)
    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))