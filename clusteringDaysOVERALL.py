#Code for Clusting & K-means analysis for EV Charging Data from 01-01-21 to 31-12-21
#PROGRESS: Done

"""
Created on
Modified on
@author: twong, amartinez
"""

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
with open("Charging Data Oct 2020 - Sep 2021.json", "r") as file:
    data = json.load(file)

# Normalize the '_items' list
items_df = pd.json_normalize(data['_items'])
visual_df = pd.json_normalize(data['_items'])

# Display DataFrame
print(items_df)
X = items_df


# %% Elbow Method to determine how many clusters

#converting to date time format
items_df['connectionTime'] = pd.to_datetime(items_df['connectionTime'])

# Extract month from 'connectionTime'
items_df['hour'] = items_df['connectionTime'].dt.hour

# Select features for clustering
X = items_df[['kWhDelivered', 'hour']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate WCSS for different values of k
wcss = []
max_clusters = 24  # Maximum number of clusters to try
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(24, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.grid(True)
plt.xticks(np.arange(1, max_clusters + 1, 1))
plt.show()
 # %% Applying the K-means algorithm

# Select features for clustering
X = items_df[['kWhDelivered', 'hour']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can adjust the number of clusters as needed
kmeans.fit(X_scaled)

# Add cluster labels to dataframe
items_df['cluster'] = kmeans.labels_

# Display the cluster centers
print("Cluster centers (x=hourOfDay,y=kWhDelivered):")
for i, center in enumerate(scaler.inverse_transform(kmeans.cluster_centers_)):
    print(f"Cluster {i}: ({center[1]}, {center[0]})")

# Display the counts of data points in each cluster
print("Counts per cluster:")
print(items_df['cluster'].value_counts())

#Percentage of data in each cluster
total_samples = len(X)
cluster_counts = items_df['cluster'].value_counts()
cluster0 = (cluster_counts[0] / total_samples) * 100
cluster1 = (cluster_counts[1] / total_samples) * 100
cluster2 = (cluster_counts[2] / total_samples) * 100
print("Percentage of data assigned to each cluster:")
print("Cluster 0:", cluster0)
print("Cluster 1:", cluster1)
print("Cluster 2:", cluster2)

# Scatter plot of kwhDelivered vs month, colored by cluster
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each cluster separately
for cluster_label in items_df['cluster'].unique():
    cluster_data = items_df[items_df['cluster'] == cluster_label]
    ax.scatter(cluster_data['hour'], cluster_data['kWhDelivered'], label=f'Cluster {cluster_label}')

# Add labels and title
ax.set_xlabel('Hour')
ax.set_ylabel('kWhDelivered')
ax.set_title('Clusters of Charging Data per Hour')
plt.legend(title='Clusters', loc='upper right')

# Display the plot
plt.show()


# %%
