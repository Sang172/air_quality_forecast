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

# AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
# AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
# s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)
s3 = s3fs.S3FileSystem()


def get_openaq_locations(bbox, date_from, date_to, retries=10, retry_delay=2):
    """Retrieves OpenAQ location IDs within a bounding box and date range.

    Input Type: bbox (str), date_from (str), date_to (str), retries (int), retry_delay (int)
    Output Type: list[str] or None
    """
    url = 'https://api.openaq.org/v3/locations'
    url = url + '?bbox=' + bbox + '&params_id=2' + '&limit=1000' + f'&date_from={date_from}' + f'&date_to={date_to}'

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()['results']

            location_ids = {x['id']: [y['id'] for y in x['sensors'] if y['parameter']['id'] == 2] for x in data}
            location_ids = [k for k, v in location_ids.items() if len(v) > 0]
            return location_ids

        except requests.exceptions.Timeout:
            logger.info(f"Attempt {attempt + 1}/{retries}: Request timed out.")
            if attempt < retries - 1:
                time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            logger.info(f"Attempt {attempt + 1}/{retries}: Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

        except KeyError as e:
            logger.info(f"Attempt {attempt + 1}/{retries}: KeyError - 'results' key not found in response.  API response may have changed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            return None

        except Exception as e:
            logger.info(f"Attempt {attempt + 1}/{retries}: An unexpected error occurred: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    logger.info(f"Failed to retrieve location IDs after {retries} attempts.")
    return None
    

def process_location(location_id, date_from, date_to):
    """Downloads, processes, and saves OpenAQ data for a given location ID.

    This function retrieves gzipped CSV files from a public S3 bucket,
    concatenates them into a single Pandas DataFrame, converts datetime
    columns to date format, and saves the result to a private S3 bucket.
    It handles temporary file storage and cleanup. It iterates through
    the date range by month.

    Input Type: location_id (str), date_from (str), date_to (str)
    Output Type: tuple (str, bool) - (location_id, success_status)
    """
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
    """Processes OpenAQ data for multiple locations concurrently.

    Input Type: location_ids (list[str]), date_from (str), date_to (str), max_workers (int)
    Output Type: None
    """
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
    """Combines CSV files from an S3 bucket, filters by 'pm25', and sorts.

    Input Type: bucket (str)
    Output Type: pd.DataFrame or None
    """
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

        if not df_list:
          logger.info("No files were combined.")
        else:
            for file_path in all_files:
                try:
                    s3.rm(file_path)
                    logger.info(f"Deleted: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

        return combined_df
    
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None


def create_coord(df):
    """Processes a DataFrame to create and format coordinate columns.

    Input Type: df (pd.DataFrame)
    Output Type: pd.DataFrame
    """
    air_quality = df.copy()
    air_quality['location'] = air_quality['location'].apply(lambda x: x.strip())
    air_quality['value'] = air_quality['value'].apply(lambda x: max(0,x)).apply(lambda x: min(100,x))
    air_quality['lat'] = air_quality['lat'].apply(lambda x: abs(x))
    air_quality['lon'] = air_quality['lon'].apply(lambda x: -abs(x))
    avg_coords = air_quality.groupby('location')[['lat', 'lon']].mean().reset_index()

    air_quality = air_quality.drop(['lat', 'lon'], axis=1) 
    air_quality = air_quality.merge(avg_coords, on='location', how='left')

    air_quality['aq_lat'] = air_quality['lat'].apply(lambda x: np.round(x,6))
    air_quality['aq_lon'] = air_quality['lon'].apply(lambda x: np.round(x,6))
    air_quality = air_quality.drop(columns=['lat','lon'])
    air_quality['aq_coord'] = air_quality.apply(lambda x: (x['aq_lat'],x['aq_lon']), axis=1)
    air_quality = air_quality.drop_duplicates(subset=['datetime', 'aq_coord'])
    return air_quality



def standardize_datetime(air_quality_df):
    """Standardizes 'datetime' so that the dataset becomes hourly data

    Input Type: air_quality_df (pd.DataFrame)
    Output Type: pd.DataFrame
    """
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
    """Resolves cases where different location names are attached to the same coordinate.

    Input Type: df (pd.DataFrame)
    Output Type: pd.DataFrame
    """
    df_consolidated = df.copy()

    if 'location' not in df_consolidated.columns:
        df_consolidated['location'] = np.nan

    most_frequent_locations = df_consolidated.groupby('aq_coord')['location'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    location_map = most_frequent_locations.to_dict()

    df_consolidated['location'] = df_consolidated['location'].fillna(df_consolidated['aq_coord'].map(location_map))

    unique_counts_after_impute = df_consolidated.groupby('aq_coord')['location'].nunique()
    condition = df_consolidated['aq_coord'].map(unique_counts_after_impute) > 1
    df_consolidated.loc[condition, 'location'] = df_consolidated.loc[condition, 'aq_coord'].map(location_map)

    return df_consolidated


def open_meteo_api_call(aq_lat, aq_lon, date_from, date_to, retries=10, retry_delay=2):
    """Retrieves weather data from the Open-Meteo Archive API.

    Input Type: aq_lat (float), aq_lon (float), date_from (str), date_to (str), retries (int), retry_delay (int)
    Output Type: pd.DataFrame or None
    """
    url1 = "https://archive-api.open-meteo.com/v1/archive?"
    url2 = f"latitude={aq_lat}&longitude={aq_lon}"
    url3 = f"&start_date={date_from}&end_date={date_to}&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=America%2FLos_Angeles"
    url = url1+url2+url3

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data['hourly'])
            df['w_lat'] = data['latitude']
            df['w_lon'] = data['longitude']
            df['w_elevation'] = data['elevation']
            df['w_coord'] = df.apply(lambda x: (x['w_lat'], x['w_lon']), axis=1)
            return df

        except requests.exceptions.Timeout:
            logger.info(f"Attempt {attempt + 1}/{retries}: Request timed out.")
            if attempt < retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            logger.info(f"Attempt {attempt + 1}/{retries}: Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.info(f"Attempt {attempt + 1}/{retries}: An unexpected error occured: {e}")
            if attempt < retries - 1:
               time.sleep(retry_delay)

    logger.info(f"Failed to retrieve data after {retries} attempts.")
    return None


def get_closest_weather(air_quality, date_from, date_to):
    """Retrieves and combines weather data for unique coordinates in air_quality.

    Input Type: air_quality (pd.DataFrame), date_from (str), date_to (str)
    Output Type: pd.DataFrame
    """
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
    """Merges air quality and weather data on coordinates and datetime.

    Filters for coordinates with >1000 entries and distances < 3 miles.

    Input Type: air_quality (pd.DataFrame), weather (pd.DataFrame)
    Output Type: pd.DataFrame
    """
    air_quality['datetime'] = pd.to_datetime(air_quality['datetime'])
    weather['time'] = pd.to_datetime(weather['time'])
    data = air_quality.merge(weather, left_on=['aq_coord','datetime'], right_on = ['aq_coord','time'], how='left')
    data = data.drop_duplicates(subset=['datetime', 'aq_coord'])
    above1000 = data['aq_coord'].value_counts()
    above1000 = above1000[above1000>1000].index
    data = data[data['aq_coord'].isin(above1000)]
    data = data.sort_values(['aq_coord','datetime']).reset_index(drop=True)
    data['lat_diff'] = data['aq_lat'] - data['w_lat']
    data['lon_diff'] = data['aq_lon'] - data['w_lon']
    data = data[(data['distance_mi']<3)]
    return data



def cyclical_encoding(df1):
    """Applies cyclical encoding to time-based features and wind direction.

    Extracts month, day of week, day of year, and hour from 'time'.
    Encodes these and 'wind_direction_10m' using sin/cos transformations.

    Input Type: df1 (pd.DataFrame)
    Output Type: pd.DataFrame
    """
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
    """Normalizes numerical features and target variables using StandardScaler.

    Saves fitted scalers to S3.

    Input Type: df (pd.DataFrame), feature_num (list[str]), target (list[str])
    Output Type: pd.DataFrame
    """
    scaler_feature = StandardScaler()
    df[feature_num] = scaler_feature.fit_transform(df[feature_num])
    with s3.open('s3://air-quality-forecast/scaler_feature.pickle','wb') as file:
        pickle.dump(scaler_feature, file)

    scaler_target = StandardScaler()
    df[target] = scaler_target.fit_transform(df[target])
    with s3.open('s3://air-quality-forecast/scaler_target.pickle','wb') as file:
        pickle.dump(scaler_target, file)
    return df



def find_valid_indices(df, location_col='aq_coord', datetime_col='time', lag_hours=23, lead_hours=24):
    """Identifies indices with sufficient consecutive hourly data for lags and leads.

    Checks for `lag_hours` of consecutive hourly data *before* and
    `lead_hours` of consecutive hourly data *after* each index, grouped by location.

    Input Type: df (pd.DataFrame), location_col (str), datetime_col (str), lag_hours (int), lead_hours (int)
    Output Type: np.ndarray (indices)
    """
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
    """Creates input (X) and output (y) arrays for LSTM from valid indices.

    Uses indices in `v` to slice `df` and construct 3D input array `X`
    (samples, timesteps, features) and 2D output array `y` (samples, timesteps).

    Input Type: v (np.ndarray), df (pd.DataFrame), target (list[str]), feature_num (list[str])
    Output Type: tuple[np.ndarray, np.ndarray] (X, y)
    """
    df = df.copy()

    features = feature_num.copy()

    d = len(v)
    X = np.full((d, 24, len(features)+1), np.nan) 
    y = np.full((d, 24), np.nan)

    all_cols = target + feature_num
    data_values = df[all_cols].values
    target_values = df[target].values

    for i, row in enumerate(v):
        start_idx = row - 23
        end_idx = row + 1
        X[i] = data_values[start_idx:end_idx]

        y_start_idx = row + 1
        y_end_idx = row + 25
        y[i] = target_values[y_start_idx:y_end_idx].squeeze()     
        
        if (i+1)%1000==0:
            logger.info(f"{i+1} th datapoint added")

    return X, y



def main():

    logger.info('Start retrieving air quality data from OpenAQ')
    location_ids = get_openaq_locations(bbox, date_from, date_to)
    process_openaq_data(location_ids, date_from, date_to)
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
    data = normalize_data(data, feature_num, target)
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
