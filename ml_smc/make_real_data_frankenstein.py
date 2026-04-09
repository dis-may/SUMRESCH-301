from pylab import *
import numpy as np
import os

N_COLS = 7 # x, y, α, ls, rs, lm, rm

robot_types = ["aggression", "fear", "love", "explorer"]

# Laptop directory 
# DATA_DIR = "/Users/maygan/Documents/notes/CS789/compsci-789/braitenberg_data/" + ROBOT_TYPE + "/" # hard-coded for laptop

# PC directory
DATA_DIR = "C:\\Users\\May\\Documents\\compsci-789\\braitenberg_data\\"

## load every .npy file in DATA_DIR and concatenate them into one big array

all_data = np.empty((0, 7))


for robot_type in robot_types:
    data_dir = DATA_DIR + robot_type + "\\"
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    data_list = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        traj_data = np.load(file_path)
        assert(traj_data.shape == (501,7))    
        data_list.append(traj_data)
    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    
    all_data = np.concatenate((all_data, data), axis=0)
    #data = data[:,3:]
    print("all_data", all_data.shape)
    N_ITS = data.shape[0]
    # print(data.shape)  # (5060100, 7) -> there are 1010 trajectories, each with 501 time steps

# print(data[-5000:-4990,:])

#all data shape (20240400, 7)
np.save("frankenstein_data.npy", all_data)
