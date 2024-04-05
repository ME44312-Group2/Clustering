# Clustering

This repository is for the Clustering portion of our group project. 

## Overview 

Research Question: What is the hourly load demand pattern (in kWh) on charging stations during the day and what are the weekly/seasonal differences over a year? 

A K-means clustring algorithm that explores the possible seasonal varitions of charging demand throughout the course of 1 year. This is accomplished by using the kwhDelivered and date based on connectionTime to analyse and capture any patterns. 

## Code Breakdown

In the first section of the code we simply extract the EV Charging data from a JSON into a Pandas Dataframe for compatability with the Pandas library.This also enables easy interaction with the data within Python using matrix indexing through the dataframe. 

We can use python to visualize our own data. We can see how many charges we have per month. 

## Next Steps

### Look at clustering for different times/patterns

    -Throughout different months (DONE)
    x-axis = 9 points for each month in dataset
        -Overall = using all data from Oct 2020 - Sep 2021

    -Throughout different times in the day  
    x-axis = 24 points for each hour of the day
        -Overall = using all data from Jan-Sep (1 graph)
        -Monthly = month-by-month analysis (12 graphs)
        
    -Throughout the week (overall and monthly breakdown) x-axis = 7 points for each day of the week
        -Overall = using all data from Jan-Sep (1 graph)
        -Monthly = month-by-month analysis (12 graphs)