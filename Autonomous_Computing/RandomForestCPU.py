#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:46:56 2019

@author: samas
"""

 # Pandas is used for data manipulation
import pandas as pd
    
#Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv('/Users/samas/Downloads/stat_train.csv')
features.head(5)

# Split the time into sub components\n",
features['time'] = pd.to_datetime(features['time'])
features['year'] = features['time'].dt.year
features['month'] = features['time'].dt.month
features['day'] = features['time'].dt.day
features['hour'] = features['time'].dt.hour
features['minute'] = features['time'].dt.minute
print('The shape of our features is:', features.shape)
# Descriptive statistics for each column
features.describe()
# Use datetime for dealing with dates
import datetime
dates = features['time']
# Import matplotlib for plotting and use magic command for Jupyter Notebooks

import matplotlib.pyplot as plt

#%matplotlib inline
# Set the style\n",
plt.style.use('fivethirtyeight')

from pandas.plotting import register_matplotlib_converters,register_matplotlib_converters
#warnings.warn(msg, FutureWarning)  

# Set up the plotting layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# CPU Utilisation
ax1.plot(dates, features['cpuutilisation'])
ax1.set_xlabel(''); ax1.set_ylabel('CPU %'); ax1.set_title('CPU Utilisation')
   
# CPU IO Wait
ax2.plot(dates, features['cpuiowait'])
ax2.set_xlabel(''); ax2.set_ylabel('CPU IO Wait (millis)'); ax2.set_title('CPU IO Wait')

# CPU Context Switches
ax3.plot(dates, features['cpucontextswitches'])
ax3.set_xlabel('Time'); ax3.set_ylabel('CPU Context Switche count'); ax3.set_title('CPU Context Switches')
   
plt.tight_layout(pad=2)
 
fig 

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['cpuutilisation'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop(['hostname','time','memoryavailable','diskreadcount','diskwritecount','diskreadtime','diskwritetime','networkopenconnections','networkreceiveerrors'], axis = 1)
    
# Saving feature names for later use
feature_list = list(features.columns)
 
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets\n",

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets\n",
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,
                                                                           random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)"
 

# The baseline predictions are the historical averages

baseline_preds = test_features[:, feature_list.index('cpuutilisation')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')
 
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
   
# Instantiate model
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels);"
  
# Use the forest's predict method on the test data\n",
predictions = rf.predict(test_features)

# Calculate the absolute errors\n",
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)\n",
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
  

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
mape = np.nan_to_num(mape)

 Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
 

rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None,
    "                               min_samples_split = 2, min_samples_leaf = 1)
 
  
# Import tools needed for visualization
from sklearn.tree import export_graphviz
"import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    
# Use dot file to create a graph\n",
(graph, ) = pydot.graph_from_dot_file('tree.dot')
    
# Write graph to a png file
graph.write_png('tree.png');
print('The depth of this tree is:', tree.tree_.max_depth)
 

# Limit depth of tree to 2 levels\n",
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features, train_labels)
   
# Extract the small tree
tree_small = rf_small.estimators_[5]
    
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)  
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')    
graph.write_png('small_tree.png');"
 

# Get numerical feature importances
importances = list(rf.feature_importances_)
   
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
  
# Sort the feature importances by most important first\n",
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
   
# Print out the feature and importances 
for pair in feature_importances:
   print('Variable: {:20} Importance: {}'.format(*pair))
 
# New random forest with only the two most important variables\n",
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
 
# Extract the two most important features
important_indices = [feature_list.index('cpuutilisation'), feature_list.index('cpucontextswitches')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)  
errors = abs(predictions - test_labels)
  
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
 
mape = 100 * (errors / test_labels)
mape = np.nan_to_num(mape)
mape = np.mean(mape)
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

# list of x locations for plotting
x_values = list(range(len(importances)))
    ,
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
  
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
  
# Axis labels and title\n",
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 
  
# Dates of training values\n",
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
hours = features[:, feature_list.index('hour')]
minutes = features[:, feature_list.index('minute')]
seconds = features[:, feature_list.index('second')]
  
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) + ' ' + str(int(hour)) + ':' + str(int(minute)) + ':' + str(int(second)) for year, month, day, hour, minute, second in zip(years, months, days, hours, minutes, seconds)
dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]
  
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
hours = test_features[:, feature_list.index('hour')]
minutes = test_features[:, feature_list.index('minute')]
seconds = test_features[:, feature_list.index('second')]

# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) + ' ' + str(int(hour)) + ':' + str(int(minute)) + ':' + str(int(second)) for year, month, day, hour, minute, second in zip(years, months, days, hours, minutes, seconds)]

# Convert to datetime objects\n",
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in test_dates]

# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions}) 
 
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# Graph labels
plt.xlabel('Date'); plt.ylabel('CPU Utilisation'); plt.title('Actual and Predicted Values');
 
