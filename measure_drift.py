 # Calculates drift and submits back to MLOps for further analysis
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
df.tail()

# Test if current distribution is different than training data set

prediction_dist_series = df.groupby(df["metrics.prediction"]).describe()["metrics.Alcohol"]["count"]
prediction_dist_series
x2, pv = chisquare([(training_distribution_percent["Poor"] * len(df))[0], \
                    (training_distribution_percent["Excellent"] * len(df))[0]],\
                   [prediction_dist_series[0], prediction_dist_series[1]])

print(x2, pv)

# Put it back into MLOps for Tracking
cdsw.track_aggregate_metrics({"chisq_x2": x2, "chisq_p": pv}, current_timestamp_ms, current_timestamp_ms, model_deployment_crn=model_deployment_crn)