import requests
from datetime import datetime, timedelta
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
import boto3
import s3fs
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import tempfile
import io
import zipfile

load_dotenv()

api_key=os.environ.get('OPENAQ_API_KEY')
headers = {"X-API-Key": api_key}
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
MODEL_FILE_NAME = 'lstm.keras'
SCALER_FILE_NAME = 'scaler.pickle'
LOCAL_MODEL_PATH = os.path.join(tempfile.gettempdir(), 'local_lstm.keras')
LOCAL_SCALER_PATH = os.path.join(tempfile.gettempdir(), 'local_scaler.pickle')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def download_from_s3(s3_file_name, local_path):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3.download_file(S3_BUCKET_NAME, s3_file_name, local_path)
        logger.info(f"Downloaded {s3_file_name} from S3 to {local_path}")
    except Exception as e:
        logger.info(f"Error downloading {s3_file_name} from S3: {e}")
        raise

download_from_s3(MODEL_FILE_NAME, LOCAL_MODEL_PATH)
download_from_s3(SCALER_FILE_NAME, LOCAL_SCALER_PATH)

model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
with open(LOCAL_SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


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
    


def get_sensor_ids(coord, hours_ago=48):
    now_utc = datetime.now(pytz.utc)
    pacific_tz = pytz.timezone('US/Pacific')
    now_pacific = now_utc.astimezone(pacific_tz)

    date_to = now_pacific.isoformat()[:10]
    date_from = (now_pacific - timedelta(hours=hours_ago)).isoformat()[:10]
    url1 = "https://api.openaq.org/v3/locations?coordinates="
    url2 = f"{coord[0]},{coord[1]}"
    url3 = "&params_id=2&radius=25000&limit=1000" + f'&date_from={date_from}' + f'&date_to={date_to}'
    base_url = url1 + url2 + url3


    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        data = response.json()['results']
        location_ids = {x['id']: ([y['id'] for y in x['sensors'] if y['parameter']['id']==2], x['distance'], (x['coordinates']['latitude'],x['coordinates']['longitude']) ) for x in data}
        location_ids = {k: v for k,v in location_ids.items() if len(v[0])>0}
        location_ids = sorted(location_ids.items(), key = lambda x: x[1][1])
        location_ids = dict(location_ids)
        sensor_ids = {v[0][0]:v[2] for k,v in location_ids.items()}
        return sensor_ids

    except requests.exceptions.RequestException as e:
        logger.info(f"OpenAQ API Error: {e}")
        return None

    except ValueError as e:
        logger.info(f"Error parsing OpenAQ response: {e}")
        return None
    


def get_sensor_measurements(sensor_id, hours_ago=48):

    now_utc = datetime.now(pytz.utc)
    pacific_tz = pytz.timezone('US/Pacific')
    now_pacific = now_utc.astimezone(pacific_tz)

    date_to = now_pacific.isoformat()
    date_from = (now_pacific - timedelta(hours=hours_ago)).isoformat()
    url1 = "https://api.openaq.org/v3/sensors/"
    url2 = f"{sensor_id}/hours?"
    url3 = f'&date_from={date_from[:10]}' + f'&date_to={date_to[:10]}' + "&limit=100"
    base_url = url1 + url2 + url3

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        data = response.json()['results']
        return data

    except requests.exceptions.RequestException as e:
        logger.info(f"OpenAQ API Error: {e}")
        return None

    except ValueError as e:
        logger.info(f"Error parsing OpenAQ response: {e}")
        return None
    

def get_closest_measurement(sensor_ids, hours_ago=48):
    
    now_utc = datetime.now(pytz.utc)
    pacific_tz = pytz.timezone('US/Pacific')
    now_pacific = now_utc.astimezone(pacific_tz)

    date_to = now_pacific.isoformat()
    date_from = (now_pacific - timedelta(hours=hours_ago)).isoformat()

    outer_break=False
    for k,v in sensor_ids.items():
        m = get_sensor_measurements(k)
        if m:
            m = [(x['period']['datetimeTo']['local'], x['value'],v) for x in m if x['period']['datetimeTo']['local']<=date_to and x['period']['datetimeTo']['local']>=date_from]
            if len(m)>10:
                outer_break = True
                break
        if outer_break:
            break
        time.sleep(0.5)
    if m is None or len(m)<=10:
        logger.infot("Could not find nearby air quality data for the last 24 hours")
        return None
    return m
        

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
    response = requests.get(url)

    df = create_dataframe(measurements)

    if response.status_code == 200:
        data = response.json()
        df1 = pd.DataFrame(data['hourly'])
        df1['w_lat'] = data['latitude']
        df1['w_lon'] = data['longitude']
        df1['w_elevation'] = data['elevation']
        df1['w_coord'] = df1.apply(lambda x: (x['w_lat'],x['w_lon']), axis=1)
        df1['time'] = pd.to_datetime(df1['time'])
        df1['distance_mi'] = df1.apply(lambda x: geodesic(x['w_coord'],coord).miles, axis=1)
        df = df.merge(df1, left_on=['time'], right_on = ['time'], how='left')
        df['lat_diff'] = df['aq_lat'] - df['w_lat']
        df['lon_diff'] = df['aq_lon'] - df['w_lon']
        df = df.drop(columns=['w_lat','w_lon','w_coord'])
        return df
    else:
        print(f"Error: {response.status_code}")


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


def normalize_data(df, feature_num):
    df[feature_num] = scaler.transform(df[feature_num])
    df[feature_num] = df[feature_num].interpolate(method='linear').bfill().ffill()
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



def load_predict(data, target, feature_num):
    X = data[target+feature_num].values
    X = np.expand_dims(X, axis=0)
    pred = model.predict(X)
    latest = data.iloc[-1]['time']
    forecasts = generate_next_24_hours(str(latest))
    forecasts = {forecasts[i]: pm25_to_aqi(pred[0][i]) for i in range(24)}
    return forecasts



def main(address: str):
    logger.info(f"Starting main process for address: {address}")

    logger.info("Geocoding address...")
    coord = geocode_address(address)
    if not coord:
        logger.error("Could not geocode the provided address.")
        return None

    logger.info(f"Geocoded coordinates: {coord}")

    logger.info("Getting nearby sensor IDs...")
    sensor_ids = get_sensor_ids(coord)
    if not sensor_ids:
        logger.error("Could not find sensor IDs for the provided coordinates.")
        return None

    logger.info(f"Found nearby sensor IDs")

    logger.info("Getting closest measurements...")
    measurements = get_closest_measurement(sensor_ids)
    if not measurements:
        logger.error("Could not find closest measurements")
        return None
    
    logger.info(f"Got closest measurements")

    logger.info("Calling Open-Meteo API...")
    data = open_meteo_api_call(measurements)
    if data is None:
        logger.error("Could not retrieve weather data.")
        return None  # No point continuing
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
    data = normalize_data(data, feature_num)
    logger.info("Feature engineering complete.")

    logger.info("Loading model and running prediction...")
    forecasts = load_predict(data, target, feature_num)
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