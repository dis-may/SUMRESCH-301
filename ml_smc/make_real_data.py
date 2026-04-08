from pylab import *
import numpy as np

N_COLS = 7 # x, y, α, ls, rs, lm, rm

# ROBOT_TYPE = "aggression"
# ROBOT_TYPE = "love"
# ROBOT_TYPE = "fear"
ROBOT_TYPE = "explorer"
DATA_DIR = "/Users/maygan/Documents/notes/CS789/compsci-789/braitenberg_data/" + ROBOT_TYPE + "/"

## load every .npy file in DATA_DIR and concatenate them into one big array
import os
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
data_list = []
for file_name in file_list:
    file_path = os.path.join(DATA_DIR, file_name)
    traj_data = np.load(file_path)
    assert(traj_data.shape == (501,7))    
    data_list.append(traj_data)
data = np.concatenate(data_list, axis=0)
#data = data[:,3:]
N_ITS = data.shape[0]
# print(data.shape)  # (5060100, 7) -> there are 1010 trajectories, each with 501 time steps

print(data[-5000:-4990,:])
np.save(f"{ROBOT_TYPE}_data.npy", data)
