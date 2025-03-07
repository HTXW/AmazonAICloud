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
import math
import sagemaker.amazon.common as smac
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

# Define data paths on S3
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)

d2 = pd.read_csv('df2.csv', index_col='Timestamp')
df = d2.iloc[:-30]
df2_last_30 = d2.iloc[-30:]

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

forecast = pipeline.predict(
    context=torch.tensor(df["target"]),
    prediction_length=30,
    num_samples=30,
)

print(df2_last_30)

forecast.mean(dim=(0, 1))

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
forecast_index = range(len(low))

plt.figure(figsize=(12, 6))
plt.plot(median, label="Median forecast")
plt.fill_between(forecast_index, low, high, alpha=0.3, label="80% prediciton interval")
plt.plot(df2_last_30['target'], label="Actual usage")
plt.title("DeepAR Model Prediction")
plt.ylabel("CPU usage [MHz]")
plt.xlabel("Time")
plt.xticks(rotation=90)
plt.legend()
plt.grid()
plt.show()

y_true = df2_last_30['target']
y_pred = median
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
ape = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
mape = np.mean(ape) * 100

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)
print('Mean Absolute Percentage Error (MAPE):', mape)