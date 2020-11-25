# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:03:57 2020

This script converts Paparazzi .data files into something PGFplots can easily read

@author: Kirk
"""
import numpy as np
import matplotlib.pyplot as plt

FOLLOWER_ID = 2
TARGET_ID = 1
INS_FACTOR = 0.0039063
input_filename = "20_09_04__13_36_32.data"
follower_output_filename = "test_follower.txt"
target_output_filename = "test_target.txt"
norm_error_output_filename = "norm.txt"

# start and end times for the data written to file ONLY
START_TIME = 0
END_TIME = 270000

# Load in .data file
parsed_data = []
with open(input_filename) as f:    
    data = f.read()
    for line in data.splitlines():
        parsed_data.append(line.split(" "))
        
parsed_data = np.asarray(parsed_data)

# Extract follower data
follower_positions = []
target_positions = []
follower_times = []
target_times = []
for i in range(len(parsed_data)):
    if (int(parsed_data[i][1]) == FOLLOWER_ID) & (parsed_data[i][2] == 'INS') & (float(parsed_data[i][0]) > START_TIME) & (float(parsed_data[i][0]) < END_TIME):
        follower_INS_value = np.array([float(parsed_data[i][3]), -float(parsed_data[i][4]), -float(parsed_data[i][5])])
        follower_position = follower_INS_value * INS_FACTOR # converting INS readout to metres        
        follower_positions.append(follower_position)
        follower_times.append(float(parsed_data[i][0]))
    
    if (int(parsed_data[i][1]) == TARGET_ID) & (parsed_data[i][2] == 'INS') & (float(parsed_data[i][0]) > START_TIME) & (float(parsed_data[i][0]) < END_TIME):
        target_INS_value = np.array([float(parsed_data[i][3]), -float(parsed_data[i][4]), -float(parsed_data[i][5])])
        target_position = target_INS_value * INS_FACTOR # converting INS readout to metres        
        target_positions.append(target_position)
        target_times.append(float(parsed_data[i][0]))

follower_positions = np.asarray(follower_positions)
follower_times = np.asarray(follower_times)

target_positions = np.asarray(target_positions)
target_times = np.asarray(target_times)

# Adjust to make both arrays have the same length
norm_error = np.linalg.norm(follower_positions - target_positions[0:-1,:], axis = 1)
all_norm_error_data = np.concatenate([follower_times.reshape([-1,1]), norm_error.reshape([-1,1])])

# Reshape the data
all_follower_data = np.concatenate([follower_times.reshape([-1,1]) - START_TIME,follower_positions], axis = 1)
all_target_data = np.concatenate([target_times.reshape([-1,1]) - START_TIME,target_positions], axis = 1)

# Process the data to look proper in X and Y
all_follower_data[:,2] = -all_follower_data[:,2]
all_target_data[:,2] = -all_target_data[:,2]

fig, axes = plt.subplots()
axes.plot(follower_times,follower_positions,label={"X","Y","Z"})
#plt.legend()
axes.plot(target_times,target_positions)

fig2, axes2 = plt.subplots()
axes2.plot(follower_times,norm_error)

# Writing the processed data to a file
with open(follower_output_filename,'w') as file_to_write:
    #for row in follower_positions:
    np.savetxt(file_to_write, all_follower_data)
    
# Writing the processed data to a file
with open(target_output_filename,'w') as file_to_write:
    #for row in follower_positions:
    np.savetxt(file_to_write, all_target_data)
    
# Writing the processed data to a file
with open(norm_error_output_filename,'w') as file_to_write:
    #for row in follower_positions:
    np.savetxt(file_to_write, all_norm_error_data)