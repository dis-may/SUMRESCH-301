from pylab import *
import torch
#from torch import nn
#from torch.utils.data import DataLoader, TensorDataset
from dsms_predictor_nn import dSMSPredictorNN

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = dSMSPredictorNN().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

RES = 11
MIN_S = 0
MAX_S = 1
MIN_M = -1.05
MAX_M = 1.05
lss = np.linspace(MIN_S, MAX_S, RES)
rss = np.linspace(MIN_S, MAX_S, RES)
lms = np.linspace(MIN_M, MAX_M, RES)
rms = np.linspace(MIN_M, MAX_M, RES)

mesh = np.meshgrid(lss, rss, lms, rms)

def sample_output(input) :
    #return(input[0], input[1], input[2], input[3])
    input_tensor = torch.from_numpy(input.reshape(1,4)).float().to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.cpu().detach().numpy().reshape(-1)

outputs = apply_along_axis(
    sample_output,
    4,
    np.stack(mesh, axis=-1)
)

# set default colormap to 'viridis'
mpl.rcParams['image.cmap'] = 'berlin'

MIN_DELTA = -0.9
MAX_DELTA = +0.9

## unspecified indices
U = -1#int(RES//2)

#print(np.shape(outputs))
NR = 2; NC = 4
## for the first row, lm=rm={MIN_M}
figure(figsize=(12,6))
suptitle(f'where unspecified, s = {lss[U]:.2f}, m = {lms[U]:.2f}')
subplot2grid((NR,NC),(0,0))
imshow(outputs[:,:,U,U,0], extent=(MIN_S,MAX_S,MIN_S,MAX_S),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('ls');ylabel('rs');title('$\Delta$ ls')

subplot2grid((NR,NC),(0,1))
imshow(outputs[:,:,U,U,1], extent=(MIN_S,MAX_S,MIN_S,MAX_S),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('ls');ylabel('rs');title('$\Delta$ rs')

subplot2grid((NR,NC),(0,2))
imshow(outputs[:,:,U,U,2], extent=(MIN_S,MAX_S,MIN_S,MAX_S),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('ls');ylabel('rs');title('$\Delta$ lm')

subplot2grid((NR,NC),(0,3))
imshow(outputs[:,:,U,U,3], extent=(MIN_S,MAX_S,MIN_S,MAX_S),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('ls');ylabel('rs');title('$\Delta$ rm')

## for the second row, ls=rs={MIN_S}
subplot2grid((NR,NC),(1,0))
imshow(outputs[U,U,:,:,0], extent=(MIN_M,MAX_M,MIN_M,MAX_M),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('rm');ylabel('lm');title('$\Delta$ ls')

subplot2grid((NR,NC),(1,1))
imshow(outputs[U,U,:,:,1], extent=(MIN_M,MAX_M,MIN_M,MAX_M),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('rm');ylabel('lm');title('$\Delta$ rs')

subplot2grid((NR,NC),(1,2))
imshow(outputs[U,U,:,:,2], extent=(MIN_M,MAX_M,MIN_M,MAX_M),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('rm');ylabel('lm');title('$\Delta$ lm')

subplot2grid((NR,NC),(1,3))
imshow(outputs[U,U,:,:,3], extent=(MIN_M,MAX_M,MIN_M,MAX_M),origin='lower',vmin=MIN_DELTA,vmax=MAX_DELTA)
xlabel('rm');ylabel('lm');title('$\Delta$ rm')


tight_layout()
show()


#model.eval()
# for 
# test_input = np.array([[0,1,0,1],
#                         [0.8664798,   3.7320678,   2.8454657,  -2.8668346]])
#     test_tensor = torch.from_numpy(test_input).float().to(device)
#     with torch.no_grad():
#         pred = model(test_tensor)
#         print(f"Test Input: \n {test_tensor.cpu().numpy()}")
#         print(f"Model Output: \n {pred.cpu().numpy()}")

