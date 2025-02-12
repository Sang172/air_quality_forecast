import os
import pandas as pd
import shutil
import requests
import pickle
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
import s3fs
import tempfile
import shutil
import calendar
import concurrent.futures
from geopy.distance import geodesic
import time
import logging
from flask import Flask, render_template, request, jsonify


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


now_utc = datetime.now(pytz.utc)
pacific_tz = pytz.timezone('US/Pacific')
now_pacific = now_utc.astimezone(pacific_tz)

date_to = datetime(now_pacific.year, now_pacific.month, 1) - timedelta(days=1)
date_from = datetime(now_pacific.year, now_pacific.month, 1) - relativedelta(months=12)
date_to = date_to.strftime("%Y-%m-%d")
date_from = date_from.strftime("%Y-%m-%d")

load_dotenv()
api_key=os.environ.get('OPENAQ_API_KEY')
headers = {"X-API-Key": api_key}
parameter = 'pm25'
bbox = "-122.6445,37.1897,-121.5871,38.2033"

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)


def get_openaq_locations(bbox, date_from, date_to):
    url = 'https://api.openaq.org/v3/locations'
    url = url + '?bbox=' + bbox + '&params_id=2' + '&limit=1000' + f'&date_from={date_from}' + f'&date_to={date_to}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()['results']

        location_ids = {x['id']: [y['id'] for y in x['sensors'] if y['parameter']['id']==2] for x in data}
        location_ids = [k for k,v in location_ids.items() if len(v)>0]
        return location_ids
    except requests.exceptions.RequestException as e:
        logger.info(f"An error occurred: {e}")
        return None
    

def process_location(location_id, date_from, date_to):

    date_from_dt = datetime.strptime(date_from, '%Y-%m-%d').date()
    date_to_dt = datetime.strptime(date_to, '%Y-%m-%d').date()
    private_bucket_path="s3://air-quality-forecast/openaq_data/"

    temp_dir_path = None
    
    try:
        temp_dir_path = tempfile.mkdtemp(prefix=f"openaq_{location_id}_")
        logger.info(f"Temporary directory created: {temp_dir_path}")

        all_dataframes = []

        current_date = date_from_dt
        while current_date <= date_to_dt:
            year = current_date.year
            month = current_date.month

            public_bucket_path = f"s3://openaq-data-archive/records/csv.gz/locationid={location_id}/year={year}/month={month:02}"

            try:
                files = s3.ls(public_bucket_path)
            except FileNotFoundError:
                current_date = datetime(year, month, calendar.monthrange(year, month)[1]) + pd.DateOffset(days=1)
                current_date = current_date.date()
                continue

            for file_path in files:
                if file_path.endswith(".csv.gz"):  
                    local_file_path = os.path.join(temp_dir_path, os.path.basename(file_path))
                    try:
                        s3.get(file_path, local_file_path)
                        try:
                            df = pd.read_csv(local_file_path, compression='gzip')
                            for col in df.columns:
                                if pd.api.types.is_datetime64_any_dtype(df[col]):
                                    df[col] = df[col].dt.date
                            all_dataframes.append(df)
                        except Exception as e:
                            logger.info(f"Error reading CSV {local_file_path}: {e}")
                            return location_id, False

                    except Exception as e:
                            logger.info(f"Error downloading file {file_path}: {e}")
                            return location_id, False

            current_date = datetime(year, month, calendar.monthrange(year, month)[1]) + pd.DateOffset(days=1)
            current_date = current_date.date()


        if all_dataframes:
            final_df = pd.concat(all_dataframes, ignore_index=True)

            output_file_path = private_bucket_path + f"{location_id}.csv"

            with s3.open(output_file_path, 'w') as f:
                final_df.to_csv(f, index=False)
            logger.info(f"Data for location {location_id} saved to {output_file_path}")
            return location_id, True
        else:
            logger.info(f"No data downloaded for location {location_id}, skipping saving.")
            return location_id, True


    except Exception as e:
        logger.info(f"An error occurred processing location {location_id}: {e}")
        return location_id, False
    finally:
        if temp_dir_path and os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path)
            logger.info(f"Temporary directory removed: {temp_dir_path}")


def process_openaq_data(location_ids, date_from, date_to, max_workers=5):

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_location, location_id, date_from, date_to)
                   for location_id in location_ids]

        for future in concurrent.futures.as_completed(futures):
            location_id, success = future.result()
            if success:
                logger.info(f"Successfully processed location: {location_id}")
            else:
                logger.info(f"Failed to process location: {location_id}")


def combine_csv_files(bucket):
    try:
        s3_path = bucket

        all_files = s3.glob(f"{s3_path}*.csv")

        if not all_files:
            logger.info(f"No CSV files found in {s3_path}")
            return pd.DataFrame() 

        df_list = []

        for file_path in all_files:
            try:
                with s3.open(file_path, 'r') as f:
                    df = pd.read_csv(f)
                    df = df[df['parameter']=='pm25']
                    df_list.append(df)
            except Exception as e:
                logger.info(f"Error reading {file_path}: {e}")
                return None


        logger.info(f"Combining {len(df_list)} dataframes")
        combined_df = pd.concat(df_list, ignore_index=True)

        if 'location_id' in combined_df.columns and 'datetime' in combined_df.columns:
           combined_df = combined_df.sort_values(['location_id', 'datetime']).reset_index(drop=True)
        return combined_df

    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None


def create_coord(df):
    air_quality = df.copy()
    air_quality['location'] = air_quality['location'].apply(lambda x: x.strip())
    air_quality['value'] = air_quality['value'].apply(lambda x: max(0,x)).apply(lambda x: min(100,x))
    air_quality['lat'] = air_quality['lat'].apply(lambda x: abs(x))
    air_quality['lon'] = air_quality['lon'].apply(lambda x: -abs(x))
    avg_coords = air_quality.groupby('location')[['lat', 'lon']].mean().reset_index()

    air_quality = air_quality.drop(['lat', 'lon'], axis=1) 
    air_quality = air_quality.merge(avg_coords, on='location', how='left')

    air_quality = air_quality.drop_duplicates()
    air_quality['aq_lat'] = air_quality['lat'].apply(lambda x: np.round(x,6))
    air_quality['aq_lon'] = air_quality['lon'].apply(lambda x: np.round(x,6))
    air_quality = air_quality.drop(columns=['lat','lon'])
    air_quality['aq_coord'] = air_quality.apply(lambda x: (x['aq_lat'],x['aq_lon']), axis=1)
    return air_quality



def standardize_datetime(air_quality_df):

    df = air_quality_df.copy()
    df = df[df['datetime'].str.contains("-08:00|-07:00", na=False)]
    df['datetime'] = df['datetime'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['datetime_rounded'] = df['datetime'].dt.round('h')

    df['time_diff'] = abs((df['datetime'] - df['datetime_rounded']).dt.total_seconds())

    df = df.sort_values(by=['location', 'datetime_rounded', 'time_diff'])

    df = df.drop_duplicates(subset=['location', 'datetime_rounded'], keep='first')
    
    df = df.drop(['time_diff', 'datetime'], axis=1)
    df = df.rename(columns={'datetime_rounded': 'datetime'})
    
    df = df.sort_values(by=['location', 'datetime']).reset_index(drop=True)
    return df


def clean_location(df):

    df_consolidated = df.copy()

    def get_most_frequent(group):
        """Helper function to get the most frequent location, handling ties."""
        mode_result = group['location'].mode()
        if not mode_result.empty:
            return mode_result.iloc[0]
        else:
            return None

    most_frequent_locations = df.groupby('aq_coord').apply(get_most_frequent, include_groups=False).to_dict()

    df_consolidated['location'] = df_consolidated['aq_coord'].map(most_frequent_locations)
    return df_consolidated


def open_meteo_api_call(aq_lat, aq_lon, date_from, date_to):
    url1 = "https://archive-api.open-meteo.com/v1/archive?"
    url2 = f"latitude={aq_lat}&longitude={aq_lon}"
    url3 = f"&start_date={date_from}&end_date={date_to}&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=America%2FLos_Angeles"
    url = url1+url2+url3
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['w_lat'] = data['latitude']
        df['w_lon'] = data['longitude']
        df['w_elevation'] = data['elevation']
        df['w_coord'] = df.apply(lambda x: (x['w_lat'],x['w_lon']), axis=1)
        return df
    else:
        logger.info(f"Error: {response.status_code}")


def get_closest_weather(air_quality, date_from, date_to):
    dfs = []
    i = 1
    for aq_lat, aq_lon in air_quality['aq_coord'].unique():
        logger.info(f"Making {i} th API call to Open Meteo...")
        df = open_meteo_api_call(aq_lat, aq_lon, date_from, date_to)
        if df is not None:
            df['aq_lat'] = aq_lat
            df['aq_lon'] = aq_lon
            df['aq_coord'] = df.apply(lambda x: (x['aq_lat'],x['aq_lon']), axis=1)
            df['distance_mi'] = df.apply(lambda x: geodesic(x['w_coord'],x['aq_coord']).miles, axis=1)
            df = df.drop(columns=['aq_lat','aq_lon'])
            dfs.append(df)
        time.sleep(10)
        i += 1
    return pd.concat(dfs)


def merge_datasets(air_quality, weather):
    air_quality['datetime'] = pd.to_datetime(air_quality['datetime'])
    weather['time'] = pd.to_datetime(weather['time'])
    data = air_quality.merge(weather, left_on=['aq_coord','datetime'], right_on = ['aq_coord','time'], how='left')
    above1000 = data['aq_coord'].value_counts()
    above1000 = above1000[above1000>1000].index
    data = data[data['aq_coord'].isin(above1000)]
    data = data.sort_values(['aq_coord','datetime']).reset_index(drop=True)
    data['lat_diff'] = data['aq_lat'] - data['w_lat']
    data['lon_diff'] = data['aq_lon'] - data['w_lon']
    # data = data[(data['distance_mi']<3)]
    return data



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
    scaler = StandardScaler()
    df[feature_num] = scaler.fit_transform(df[feature_num])
    with s3.open('s3://air-quality-forecast/scaler.pickle','wb') as file:
        pickle.dump(scaler, file)
    return df


def find_valid_indices(df, location_col='aq_coord', datetime_col='time', lag_hours=23, lead_hours=24):

    df = df.copy()

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.sort_values(by=[location_col, datetime_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['time_diff'] = df.groupby(location_col)[datetime_col].diff().dt.total_seconds() / 3600.0

    df['valid_lag'] = (
        df.groupby(location_col)['time_diff']
        .rolling(window=lag_hours, min_periods=lag_hours)
        .apply(lambda x: (x == 1.0).all(), raw=True)
        .fillna(False)
        .reset_index(level=0, drop=True)
        .astype(bool)
    )

    def check_consecutive(series, window_size):
      """Helper function for lead check."""
      reversed_series = series.iloc[::-1]
      rolled = reversed_series.rolling(window=window_size, min_periods=window_size).apply(lambda x: (x==1).all(), raw=True)
      return rolled.iloc[::-1].fillna(False)

    df['valid_lead'] = df.groupby(location_col)['time_diff'].transform(check_consecutive, window_size=lead_hours+1).astype(bool)

    valid_indices = df.index[df['valid_lag'] & df['valid_lead']]
    return valid_indices.to_numpy()

def create_lstm_input_output(
    v: np.ndarray,
    df: pd.DataFrame,
    target: list[str],
    feature_num: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    df = df.copy()

    features = feature_num.copy()

    d = len(v)
    X = np.full((d, 24, len(features)+1), np.nan) 
    y = np.full((d, 24), np.nan)

    for i, row in enumerate(v):
        start_idx = row - 23
        end_idx = row + 1
        data_slice = df[target+features].iloc[start_idx:end_idx].values
        X[i] = data_slice

        y_start_idx = row + 1
        y_end_idx = row + 25
        y_slice = df[target].iloc[y_start_idx:y_end_idx]
        y[i] = y_slice.values.squeeze()        
        if (i+1)%1000==0:
            logger.info(f"{i+1} th datapoint added")

    return X, y



def main():

    logger.info('Start retrieving air quality data from OpenAQ')
    location_ids = get_openaq_locations(bbox, date_from, date_to)
    process_openaq_data(location_ids[10:12], date_from, date_to)
    logger.info('OpenAQ air quality data retrieval complete')

    logger.info('Start prcessing OpenAQ air quality data')
    bucket = 's3://air-quality-forecast/openaq_data/'
    air_quality = combine_csv_files(bucket)
    air_quality = create_coord(air_quality)
    air_quality = standardize_datetime(air_quality)
    air_quality = clean_location(air_quality)
    with s3.open('s3://air-quality-forecast/air_quality.pickle', 'wb') as f:
        pickle.dump(air_quality, f)
    logger.info('Prcessing OpenAQ air quality data complete, saved to AWS S3')


    logger.info('Start retrieving weather data from Open Meteo')
    weather = get_closest_weather(air_quality, date_from, date_to)
    with s3.open('s3://air-quality-forecast/weather.pickle', 'wb') as f:
        pickle.dump(weather, f)
    logger.info('Open Meteo weather data retrieval complete, saved to AWS S3')


    logger.info('Start Feature engineering')
    data = merge_datasets(air_quality, weather)
    id_columns = ['aq_coord','time']
    target = ['value']
    feature_num = ['aq_lat', 'aq_lon', 'lat_diff', 'lon_diff', 'distance_mi', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'cloud_cover', 'w_elevation', 'wind_speed_10m', 'wind_direction_10m']
    data = data[id_columns+target+feature_num]
    data = cyclical_encoding(data)
    feature_num.pop(-1)
    feature_num += ['month_sin','month_cos','hour_sin','hour_cos','dayofweek_sin','dayofweek_cos','dayofyear_sin','dayofyear_cos','wind_direction_10m_cos','wind_direction_10m_sin']
    data = normalize_data(data, feature_num)
    logger.info('Feature engineering complete')


    logger.info('Start reshaping data for LSTM training')
    valid_indices = find_valid_indices(data)
    X, y = create_lstm_input_output(valid_indices, data, target, feature_num)
    with s3.open('s3://air-quality-forecast/input_X.pickle', 'wb') as f:
        pickle.dump(X, f)
    with s3.open('s3://air-quality-forecast/output_y.pickle', 'wb') as f:
        pickle.dump(y, f)
    logger.info('Reshaping complete')

if __name__ == "__main__":
    main()
