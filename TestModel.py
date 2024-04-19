# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
import matplotlib.pyplot as plt

# %% Code for changing json into pandas dataframe

# Load JSON data
with open("Aug2021OutOfSample_JPL.json", "r") as file:
    data = json.load(file)

# Normalize the '_items' list
items_df = pd.json_normalize(data['_items'])
visual_df = pd.json_normalize(data['_items'])

# Display DataFrame
print(items_df)
X = items_df

target_month = 8
title = "Clusters of Charging Data Per Hour in August 2021 Out of Sample"

#converting to date time format
items_df['connectionTime'] = pd.to_datetime(items_df['connectionTime'])

# Extract month from 'connectionTime'
items_df['hour'] = items_df['connectionTime'].dt.hour
items_df = items_df[(items_df['connectionTime'].dt.month == target_month)]
# Select features for clustering
X = items_df[['kWhDelivered', 'hour']]

#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the existing centroids
existing_centroids = np.array([
    [6.007717060276247, 17.542005420054203],
    [ 12.419396946564886, 2.175572519083971],
    [ 37.19448476666666, 15.23]
])

# Define the existing centroids with scaling
existing_centroids_scaled = scaler.transform(existing_centroids)

# Compute distances between each data point and each scaled centroid
distances = np.sqrt(((X_scaled[:, np.newaxis] - existing_centroids_scaled)**2).sum(axis=2))

# Assign each data point to the nearest centroid
assigned_clusters = np.argmin(distances, axis=1)
print(assigned_clusters)
# Assign the cluster labels to your dataframe
items_df['cluster'] = assigned_clusters

# Display the cluster centers
print("Assigned Cluster Centers:")
print("Cluster x: (month, kWh)")
for i, center in enumerate(existing_centroids):
    print(f"Cluster {i}: ({center[1]}, {center[0]})")

# Display the counts of data points in each cluster
print("Counts per cluster:")
print(items_df['cluster'].value_counts())

total_samples = len(X)
# Percentage of data in each cluster
cluster_counts = items_df['cluster'].value_counts()
cluster_percentages = (cluster_counts / total_samples) * 100
print("Percentage of data assigned to each cluster:")
print(cluster_percentages)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for cluster_label in items_df['cluster'].unique():
    cluster_data = items_df[items_df['cluster'] == cluster_label]
    ax.scatter(cluster_data['hour'], cluster_data['kWhDelivered'], label=f'Cluster {cluster_label}')

ax.set_xlabel('Hour')
ax.set_ylabel('kWhDelivered')
ax.set_title(title)
plt.legend(title='Clusters', loc='upper right')
plt.show()


#%%