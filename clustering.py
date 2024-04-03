#Code for Clusting & K-means analysis for EV Charging Data from 01-01-21 to 31-12-21

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

# %% Code for changing json into pandas dataframe

# Load JSON data
with open("Charging Data 2021.json", "r") as file:
    data = json.load(file)

# Normalize the '_items' list
items_df = pd.json_normalize(data['_items'])

# Check if 'userInputs' column exists
if 'userInputs' in items_df.columns:
    # Explode the 'userInputs' column
    items_df = items_df.explode('userInputs').reset_index(drop=True)
    # Concatenate DataFrame with expanded 'userInputs' column
    items_df = pd.concat([items_df.drop(columns=['userInputs']), items_df['userInputs'].apply(pd.Series)], axis=1)

# Display DataFrame
print(items_df)
X = items_df

 # %% Applying the K-means algorithm
# Select the number of clusters #
# Here we selected three to proceed. Normally you need to decide on it through some analysis.   
k = 3

# Initialize centroids - samples k many data points randomly as the initial centroids 
centroids = X.sample(k, random_state=1)


# %%
