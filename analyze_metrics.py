 # Perfoms custom analytics on desired metrics. 

import cdsw, time, os

import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

# Define our uqique model deployment id
model_deployment_crn = "crn:cdp:ml:us-west-1:12a0079b-1591-4ca0-b721-a446bda74e67:workspace:ec3efe6f-c4f5-4593-857b-a80698e4857e/d5c3fbbe-d604-4f3b-b98a-227ecbd741b4"


# Define our training distribution for 
training_distribution_percent = pd.DataFrame({"Excellent": [0.50], "Poor": [0.50]})
training_distribution_percent

current_timestamp_ms = int(round(time.time() * 1000))

known_metrics = cdsw.read_metrics(model_deployment_crn=model_deployment_crn,
            start_timestamp_ms=0,
            end_timestamp_ms=current_timestamp_ms)

df = pd.io.json.json_normalize(known_metrics["metrics"])


# Do some conversions & Calculations
df['startTimeStampMs'] = pd.to_datetime(df['startTimeStampMs'], unit='ms')
df['endTimeStampMs'] = pd.to_datetime(df['endTimeStampMs'], unit='ms')
df["processing_time"] = (df["endTimeStampMs"] - df["startTimeStampMs"]).dt.microseconds * 1000

non_agg_metrics = df.dropna(subset=["metrics.prediction"])
non_agg_metrics.tail()

# Visualize the processing time
non_agg_metrics.plot(kind='line', x='predictionUuid', y='processing_time')

# Visualize the output distribution
prediction_dist_series = non_agg_metrics.groupby(non_agg_metrics["metrics.prediction"]).describe()["metrics.Alcohol"]["count"]
prediction_dist_series.plot("bar")


# Visualize chi squared from my bi-hourly run
chi_sq_metrics = df.dropna(subset=["metrics.chisq_x2"])

chi_sq_metrics.plot(kind='line', x='endTimeStampMs', y=['metrics.chisq_x2', 'metrics.chisq_p'])
