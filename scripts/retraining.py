import os
from datetime import datetime
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

import logging

from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import transformers
import io
import boto3
import pandas as pd
from pathlib import Path
import yaml


# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info(config)



def get_data(file_name):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=config["SOURCE_BUCKET"], Key=f'data/{file_name}')
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df


def main():    
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["MLFLOW_S3_ENDPOINT_URL"]
    os.environ["AWS_ACCESS_KEY_ID"] = config["S3_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config["S3_SECRET_KEY"]
    
    mlflow.set_tracking_uri(config["MLFLOW_URL"])
    client = MlflowClient()
    experiment = client.search_experiments(filter_string=f"name = '{config["EXPIRIMENT_NAME"]}'")[0]
    experiment_id = experiment.experiment_id

    run_name = 'Run time: ' + ' ' + str(datetime.now())

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8,enable_categorical=True)
        model.fit(X_train, y_train)
      
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path=config["MODEL_NAME"]
        )
    
      
        y_forecast_train = model.predict(X_train)
        train_metric = sklearn.metrics.mean_absolute_percentage_error(y_train, y_forecast_train)
        mlflow.log_metrics({'train_MAPE' : train_metric})
    
        y_forecast = model.predict(X_test)
        test_metric = sklearn.metrics.mean_absolute_percentage_error(y_test, y_forecast)
        mlflow.log_metrics({'test_MAPE' : test_metric})
        

if __name__ == "__main__":
    main()