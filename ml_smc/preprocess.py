from pylab import *

#BASE_NAME = "bs"
# BASE_NAME = "love"
# BASE_NAME = "aggression"
# BASE_NAME = "fear"
# BASE_NAME = "explorer"
BASE_NAME = "frankenstein"

DT = 0.01

raw = np.load(f'{BASE_NAME}_data.npy')

#raw = raw[:5000,:]

# dur = 5 
# dur_its = int(15 // 0.01)
# traj_i = 0
# start = traj_i*dur_its
# end = (traj_i+1)*dur_its
# plot(raw[start:end,0],raw[start:end,1], 'o-',label='trajectory')
# show()
# quit()


inputs          = raw[:-1, :]
correct_outputs = diff(raw, axis=0) 
#correct_outputs[:] /= DT



# remove every 500th row to avoid trajectory discontinuities
inputs = np.delete(inputs, np.s_[500::501], axis=0)
correct_outputs = np.delete(correct_outputs, np.s_[500::501], axis=0)

# plot(inputs[:,0])
# plot(correct_outputs[:,0])
# show()
# quit()

inputs          = raw[:-1, [3,4,5,6]]
correct_outputs = raw[1:, [3,4,5,6]]

# set default markersize to 1
mpl.rcParams['lines.markersize'] = 1

rows = shape(correct_outputs)[0]
# plot([0,]*rows, correct_outputs[:,0], '.', label='ls')
# plot([1,]*rows, correct_outputs[:,1], '.', label='rs')
# plot([2,]*rows, correct_outputs[:,2], '.', label='lm')
# plot([3,]*rows, correct_outputs[:,3], '.', label='rm')
# show()

print(shape(inputs))
print(shape(correct_outputs))

print(inputs[-4:])
print(correct_outputs[-4:])

np.save(f'{BASE_NAME}_inputs.npy', inputs)
np.save(f'{BASE_NAME}_correct_outputs.npy', correct_outputs)
quit()