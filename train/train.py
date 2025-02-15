import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
from dotenv import load_dotenv
import s3fs
import os
import logging



load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

SCALER_TARGET_FILE_NAME = 'scaler_target.pickle'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


logger.info("Read features")
with s3.open('s3://air-quality-forecast/input_X.pickle', 'rb') as file:
    X = pickle.load(file)

logger.info("Read target")
with s3.open('s3://air-quality-forecast/output_y.pickle', 'rb') as file:
    y = pickle.load(file)

logger.info("Read target scaler")
with s3.open('s3://air-quality-forecast/scaler_target.pickle', 'rb') as file:
    scaler_target = pickle.load(file)

def create_lstm_model(input_shape, output_shape):

    model = Sequential()
    model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, activation='tanh'))
    model.add(Dense(units=output_shape))

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test):


    y_pred = model.predict(X_test)
    mean = scaler_target.mean_
    std = scaler_target.scale_
    y_test_flat = y_test.flatten() * std + mean
    y_pred_flat = y_pred.flatten() * std + mean
    
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return rmse


def main():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52190)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]

    model = create_lstm_model(input_shape, output_shape)
    # logger.info(model.summary())

    logger.info("Start training")
    history = train_model(model, X_train, y_train, epochs=80, batch_size=32, validation_split=0.2)

    rmse = evaluate_model(model, X_test, y_test)

    temp_path = '/tmp/model.keras'
    model.save(temp_path)
    with open(temp_path, 'rb') as f:
        model_bytes = f.read()
    s3_path = 's3://air-quality-forecast/lstm.keras'
    with s3.open(s3_path, 'wb') as s3_file:
        s3_file.write(model_bytes)
    logger.info("Model saved to S3")


if __name__ == "__main__":
    main()