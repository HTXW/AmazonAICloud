import warnings
warnings.filterwarnings('ignore')

# Set up the S3 bucket and prefix for SageMaker
bucket = 'workload-prediction-v2'
prefix = 'sagemaker/test'

# Set up IAM role and SageMaker session
import sagemaker.predictor
import boto3
import s3fs
import re
from sagemaker import get_execution_role
import json
import math
from os import path
import sagemaker.amazon.common as smac

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

# Define data paths on S3
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)

# Setting up the SageMaker DeepAR container
region = sagemaker_session.boto_region_name
image_name = sagemaker.image_uris.retrieve("forecasting-deepar", region)
if image_name is None:
    raise ValueError(f"Container image not found for the region: {region}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import time
import json
import glob

df2 = pd.read_csv('df2.csv', index_col='Timestamp')
df3 = pd.read_csv('df3.csv', index_col='Timestamp')

df2.head()

df3.head()

freq = "1min"
context_length = 30
prediction_length = 30

def series_to_obj(ts, cat=None):
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))

time_series_test=[]
vm_index_range = df2['VM'].unique()
for i in vm_index_range:
    newseries = df2[df2['VM'] == i]['target']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_test.append(newseries)

time_series_training=[]
vm_index_range = df2['VM'].unique()
for i in vm_index_range:
    newseries = df2[df2['VM'] == i]['target']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_training.append(newseries[:-prediction_length])

time_series_test_pro=[]
vm_index_range = df3['VM'].unique()
for i in vm_index_range:
    newseries = df3[df3['VM'] == i]['CPU capacity provisioned [MHZ]']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_test_pro.append(newseries)

print(len(time_series_test), len(time_series_training), len(time_series_test_pro))

s3filesystem = s3fs.S3FileSystem()

encoding = "utf-8"

with s3filesystem.open(s3_data_path + "/test/test_data.json", 'wb') as fp:
    for ts in time_series_test:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

with s3filesystem.open(s3_data_path + "/train/train_data.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=image_name,
    role=role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    base_job_name='test-demo-deepar',
    output_path="s3://" + s3_output_path
)

hyperparameters = {
    "time_freq": freq,
    "context_length": context_length,
    "prediction_length": prediction_length,
    "num_cells": "64",
    "num_layers": "3",
    "likelihood": "gaussian",
    "epochs": "30",
    "mini_batch_size": "32",
    "learning_rate": "0.01",
    "dropout_rate": "0.05",
    "early_stopping_patience": "10"
}

estimator.set_hyperparameters(**hyperparameters)

data_channels = {
    "train": "s3://{}/train/".format(s3_data_path),
    "test": "s3://{}/test/".format(s3_data_path)
}

estimator.fit(inputs=data_channels)

job_name = estimator.latest_training_job.name
print(job_name)
endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=job_name,
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    image_uri=image_name,
    role=role
)

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def set_prediction_parameters(self, freq, prediction_length):
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"]):
        prediction_times = [x.index[-1] + pd.Timedelta(1, unit=x.index.freq.name) for x in ts]
        req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req, initial_args={"ContentType": "application/json"})
        return self.__decode_response(res, prediction_times, encoding)

    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        instances = [series_to_obj(ts[k], cat[k] if cat else None) for k in range(len(ts))]
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)

    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range(len(prediction_times)):
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df

predictor = DeepARPredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)

new_time_series_training = []
for ts in time_series_training:
    new_time_series_training.append(ts.asfreq('T'))

new_time_series_test = []
for ts in time_series_test:
    new_time_series_test.append(ts.asfreq('T'))

new_time_series_test_pro = []
for ts in time_series_test_pro:
    new_time_series_test_pro.append(ts.asfreq('T'))

list_of_df  = predictor.predict(new_time_series_training[1:2])
actual_data = new_time_series_test[1:2]
actual_provisioned = new_time_series_test_pro[1:2]


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(12,6))

for k in range(len(list_of_df)):
    actual_data[k][-prediction_length-context_length:].plot(label='Actual',linewidth = 2.5)
    p10 = list_of_df[k]['0.1']
    p90 = list_of_df[k]['0.9']
    plt.fill_between(p10.index, p10, p90, color='grey', alpha=0.3, label='80% Confidence Interval')
    list_of_df[k]['0.5'].plot(label='Prediction Median', color = 'blue', linewidth = 2.5)
    (list_of_df[k]['0.9']+100).plot(label='Suggested Provision', color = 'green', linewidth = 2.5)

    plt.title("DeepAR Model Prediction")
    plt.ylabel("CPU usage [MHz]")
    plt.xlabel("Time")
    plt.yticks()
    plt.xticks()
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_true = []
y_pred = []

for k in range(len(list_of_df)):
    y_true.extend(actual_data[k][-(prediction_length+context_length):].tolist())
    y_pred.extend(list_of_df[k]['0.5'].tolist())

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
ape = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
mape = np.mean(ape) * 100

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)