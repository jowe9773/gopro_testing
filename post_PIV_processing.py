#post_PIV_processing.py

"""This file contains the tools needed to deal with the PIV data after it has been created in PIVlab"""

#import neccesary packages and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from file_managers import FileManagers

#instantiate classes
fm = FileManagers()


# Replace 'filename.txt' with your file's name
filename = fm.load_fn("Select a PIV file")


# Read the CSV file into a DataFrame
df = pd.read_csv(filename, skiprows=2)


#instantiate counter variables
x_num = None
rows_before_change = 0
previous_value = df.loc[0, "x [m]"]

#cound number of rows before change in x value
for index, row in df.iterrows():
    current_value = row["x [m]"]
    
    # Check if the value in the target column has changed
    if current_value != previous_value:
        break
    
    rows_before_change += 1
    previous_value = current_value

#initialze counter variables
y_num = None
previous_value = None
change_count = 0

#count number of changes in x value
for index, row in df.iterrows():
    current_value = row["x [m]"]
    
    # Check if the value in the target column has changed
    if current_value != previous_value:
        change_count += 1
    
    previous_value = current_value

#HERE ARE THE VARIABLES
x_origin = df.loc[0, "x [m]"]
y_origin = df.loc[0, "y [m]"]

x_num = rows_before_change
y_num = change_count

x_size = df.loc[x_num, "x [m]"] - df.loc[0, "x [m]"]
y_size = df.loc[1, "y [m]"] - df.loc[0, "y [m]"]

print("X origin: ", x_origin)
print("Y origin: ", y_origin)
print("x_num: ", x_num)
print("y_num: ", y_num)
print("x_size: ", x_size)
print("y_size: ", y_size)

#save u veloxity to a 2D numpy array
array_3d = np.empty((y_num, x_num, 0))

#Extract values from the 3rd column (Column3)
values = df["u [m/s]"].values

# Reshape the values into a 2D NumPy array
array_2d = values.reshape(y_num, x_num)

# First, expand the 2D array to have a new axis (axis=2)
expanded_array = np.expand_dims(array_2d, axis=2)

# Append the expanded array along the 3rd dimension of the 3D array
array_3d = np.concatenate((array_3d, expanded_array), axis=2)