from pylab import *
import numpy as np
import os

N_COLS = 7 # x, y, α, ls, rs, lm, rm

# robot_types = ["aggression", "love"] 
# robot_types = ["fear", "explorer"] 
# robot_types = ["aggression", "fear"] 
# robot_types = ["aggression", "love", "fear", "explorer"]
robot_types = ["fear", "love"]

# Laptop directory 
DATA_DIR = "/Users/maygan/Documents/notes/CS789/compsci-789/braitenberg_data/"  # hard-coded for laptop

## load every .npy file in DATA_DIR and concatenate them into one big array

all_data = np.empty((0, 7))

minis = {} # map robot type to the minimum value
maxis = {} # map robot type to the max value
sensors = [float('inf'), float('-inf')]
motors = [float('inf'), float('-inf')]

for robot_type in robot_types:
    data_dir = DATA_DIR + robot_type + "/"
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    data_list = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        traj_data = np.load(file_path)
        assert(traj_data.shape == (501,7))    
        data_list.append(traj_data)
    data = np.concatenate(data_list, axis=0)

    minis[robot_type] = np.min(data, axis=0)
    maxis[robot_type] = np.max(data, axis=0)
    sensors[0] = min(minis[robot_type][3], minis[robot_type][4], sensors[0])
    sensors[1] = max(maxis[robot_type][3], maxis[robot_type][4], sensors[1])
    motors[0] = min(minis[robot_type][5], minis[robot_type][6], motors[0])
    motors[1] = max(maxis[robot_type][5], maxis[robot_type][6], motors[1])

    print(data.shape)
    
    all_data = np.concatenate((all_data, data), axis=0)
    #data = data[:,3:]
    print("all_data", all_data.shape)
    N_ITS = data.shape[0]
    # print(data.shape)  # (5060100, 7) -> there are 1010 trajectories, each with 501 time steps

# print(data[-5000:-4990,:])

#all data shape (20240400, 7)
np.save(f"frankenstein_{'_'.join(robot_types)}_data.npy", all_data)

print(minis)
print(maxis)
print("Sensor range", sensors)
print("Motor range", motors)

# Sensor range [0.0, 0.9938365114922629]
# Motor range [-0.8353972513197995, 0.9938365114922629]
