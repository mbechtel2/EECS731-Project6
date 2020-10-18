################################################################################
#
# File: project6
# Author: Michael Bechtel
# Date: October 19, 2020
# Class: EECS 731
# Description: Use anomaly detection models to find known anomalies in a dataset 
#               for machine temperature readings.
# 
################################################################################

# Python imports
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Create necessary sklearn objects
minMax = MinMaxScaler()
isoForest = IsolationForest(random_state=0)
localOutlier = LocalOutlierFactor()
dbscan = DBSCAN()
oneClassSVM = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

# Read the raw dataset
raw_dataset = pd.read_csv("../data/raw/machine_temperature_system_failure.csv")

# Extract the machine temperature values into two new columns
#   One for the raw values and one for min-max normalized values
values_dataset = raw_dataset["value"].values.reshape(-1,1)
values_normalized = minMax.fit_transform(values_dataset)
x_range = range(len(values_dataset)) # Used for plotting results

# Create a save a new dataset with the raw and normalized columns
new_dataset = pd.DataFrame({"Raw Values":values_dataset.squeeze(),"Normalized Values":values_normalized.squeeze()})
new_dataset.to_csv("../data/processed/values_dataset.csv")

# Uncomment to see scatter graph of original dataset without predictions
#plt.scatter(times_dataset, values_dataset["value"])
#plt.show()

################################################################################
# Raw dataset values
################################################################################

# Predict the anomalies with all models on the raw value column
IFresults = isoForest.fit_predict(values_dataset)
DBresults = dbscan.fit_predict(values_dataset)
LOresults = localOutlier.fit_predict(values_dataset)
SVMresults = oneClassSVM.fit_predict(values_dataset)

# Plot the results for each of the models
_,graphs = plt.subplots(2,2)

graphs[0,0].scatter(x_range, raw_dataset["value"], c=IFresults)
graphs[0,0].set_title("Isolation Forest")

graphs[0,1].scatter(x_range, raw_dataset["value"], c=DBresults)
graphs[0,1].set_title("DBSCAN")

graphs[1,0].scatter(x_range, raw_dataset["value"], c=LOresults)
graphs[1,0].set_title("Local Outlier Factor")

graphs[1,1].scatter(x_range, raw_dataset["value"], c=SVMresults)
graphs[1,1].set_title("One-Class SVM")

# Save the results
plt.gcf().set_size_inches((12.80,7.20), forward=False)
plt.savefig("../visualizations/raw_results.png", bbox_inches='tight', dpi=100)
#plt.show()

################################################################################
# Min-max Normalized dataset values
################################################################################

# Predict the anomalies with all models on the normalized value column
IFresults = isoForest.fit_predict(values_normalized)
DBresults = dbscan.fit_predict(values_normalized)
LOresults = localOutlier.fit_predict(values_normalized)
SVMresults = oneClassSVM.fit_predict(values_normalized)

# Plot the results for each of the models
_,graphs = plt.subplots(2,2)

graphs[0,0].scatter(x_range, raw_dataset["value"], c=IFresults)
graphs[0,0].set_title("Isolation Forest")

graphs[0,1].scatter(x_range, raw_dataset["value"], c=DBresults)
graphs[0,1].set_title("DBSCAN")

graphs[1,0].scatter(x_range, raw_dataset["value"], c=LOresults)
graphs[1,0].set_title("Local Outlier Factor")

graphs[1,1].scatter(x_range, raw_dataset["value"], c=SVMresults)
graphs[1,1].set_title("One-Class SVM")

# Save the results
plt.gcf().set_size_inches((12.80,7.20), forward=False)
plt.savefig("../visualizations/normalized_results.png", bbox_inches='tight', dpi=100)
#plt.show()