#Code for Clusting & K-means analysis for EV Charging Data from 01-01-21 to 31-12-21
#Progress: Done

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
import calendar

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
# %%
# Making a plot to check how many charges we have per month
visual_df['connectionTime'] = pd.to_datetime(visual_df['connectionTime'])

# Extracting month from 'connectionTime' and converting numeric month to month names
visual_df['month_name'] = visual_df['connectionTime'].dt.month_name()

# Counting the number of data points for each month
monthly_counts = visual_df['month_name'].value_counts().sort_index()

# Reindexing to ensure months are in order
months_in_order = [
    'October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September'
]
monthly_counts = monthly_counts.reindex(months_in_order, fill_value=0)

# Plotting the data
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Data Points for Each Month from October 2020 - September 2021')
plt.xlabel('Month')
plt.ylabel('Number of Data Points')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%