import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
#from dotenv import load_dotenv
#import s3fs
import os
import logging
import keras_tuner as kt
import argparse
import boto3
import tempfile


# load_dotenv()
# AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY')
# AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
# S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
# s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def load_data_from_s3(bucket, key):
    """Loads pickled data from an S3 object.

    Input Type: bucket (str), key (str)
    Output Type: object (data loaded from pickle)
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pickle.load(obj['Body'])


def save_model_to_s3(model, bucket, key):
    """Saves a Keras model to an S3 bucket.

    Uses a temporary directory to save the model locally before uploading.

    Input Type: model (keras.Model), bucket (str), key (str)
    Output Type: None
    """
    s3 = boto3.resource('s3')
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'lstm.keras')
        model.save(temp_path)
        s3.Bucket(bucket).upload_file(temp_path, key)


def build_model(hp, input_shape, output_shape):
    """Builds and compiles a Keras LSTM model with hyperparameter tuning.

    Input Type: hp (keras_tuner.HyperParameters), input_shape (tuple), output_shape (int)
    Output Type: keras.Model
    """
    model = Sequential()

    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=128, step=32),
            activation='tanh',
            return_sequences=True if i < hp.get('num_lstm_layers') - 1 else False,
            input_shape=input_shape if i == 0 else None
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    model.add(Dense(units=output_shape))

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model



def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
    """Trains the given Keras model with early stopping.

    Input Type: model (keras.Model), X_train (np.ndarray), y_train (np.ndarray), epochs (int), batch_size (int), validation_split (float)
    Output Type: keras.callbacks.History
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1 
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stopping]
    )
    return history


def evaluate_model(model, X_test, y_test, scaler_target):
    """Evaluates the model's performance using RMSE.

    Input Type: model (keras.Model), X_test (np.ndarray), y_test (np.ndarray), scaler_target (StandardScaler)
    Output Type: float (RMSE)
    """
    y_pred = model.predict(X_test)
    mean = scaler_target.mean_
    std = scaler_target.scale_
    y_test_flat = y_test.flatten() * std + mean
    y_pred_flat = y_pred.flatten() * std + mean
    
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return rmse



def main(args):

    bucket = args.bucket
    epochs = args.epochs
    batch_size = args.batch_size

    logger.info(f"Loading data from S3 bucket: {bucket}")
    X = load_data_from_s3(bucket, 'input_X.pickle')
    y = load_data_from_s3(bucket, 'output_y.pickle')
    scaler_target = load_data_from_s3(bucket, 'scaler_target.pickle')
    logger.info("Data loaded successfully.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]



    num_gpus = int(args.num_gpus) if args.num_gpus else 0
    if num_gpus > 0:
        logger.info(f"Num GPUs Available: {num_gpus}")
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPUs Available: {gpus}")
        if gpus:
          try:
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
          except RuntimeError as e:
            logger.error(e)
    else:
        logger.info("No GPUs detected. Training will proceed on CPU.")



    tuner = kt.BayesianOptimization(
    lambda hp: build_model(hp, input_shape, output_shape),
    objective='val_loss',
    max_trials=15,
    executions_per_trial=2,
    directory= args.output_data_dir,
    project_name='lstm_tuning',
    overwrite=True
    )

    logger.info("Start tuning")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    logger.info("Start training best model")
    history = train_model(model, X_train, y_train, epochs, batch_size, validation_split=0.2)

    rmse = evaluate_model(model, X_test, y_test, scaler_target)

    logger.info(f"Saving model to {args.model_dir}")
    save_model_to_s3(model, bucket, 'lstm.keras')
    logger.info("Model saved to S3")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bucket', type=str, default = os.environ.get('S3_BUCKET_NAME'))

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--num_gpus', type=int, default=os.environ.get('SM_NUM_GPUS'))

    args = parser.parse_args()

    main(args)