from pylab import *
import numpy as np

N_ITS = 10000
N_COLS = 7 # x, y, α, ls, rs, lm, rm
data = np.zeros((N_ITS, N_COLS), dtype=np.float32)

noise_level = 0.01

for row in range(N_ITS):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    α = np.random.uniform(0, 2*np.pi)
    ls = np.random.uniform(0, 1)
    rs = np.random.uniform(0, 1)
    lm = rs - ls + np.random.uniform(-noise_level, noise_level)
    rm = ls - rs + np.random.uniform(-noise_level, noise_level) + 1.0
    data[row,:] = [x,y,α,ls,rs,lm,rm]  # Replace x,y,α,ls,rs,lm,rm with actual values

for row in range(1,N_ITS):
    prev_ls = data[row-1,3]
    prev_rs = data[row-1,4]
    data[row,3] = data[row-1,3] + np.random.uniform(-0.01,0.01)
    data[row,4] = data[row-1,4] + np.random.uniform(-0.01,0.01)
    data[row,5] = data[row-1,5] + np.random.uniform(-0.01,0.01)
    data[row,6] = data[row-1,6] + np.random.uniform(-0.01,0.01)
    #data[row,5] = prev_rs - prev_ls + np.random.uniform(-noise_level, noise_level)
    #data[row,6] = prev_ls - prev_rs + np.random.uniform(-noise_level, noise_level) 

print(data[:,3:])
np.save("bs_data.npy", data)
