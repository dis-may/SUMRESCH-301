from dsms_predictor_nn import dSMSPredictorNN
from pylab import *
import torch

from pathlib import Path

def get_model(vehicle_type):
    """
    Takes in a vehicle type
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")

    model = dSMSPredictorNN().to(device)
    base_path = Path(__file__).parent.resolve() 
    # Construct a path to a data file relative to that directory
    file_path = base_path / "ml_models" / vehicle_type.name.lower() + "_" + "model.pth"
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    return model

def get_motor_values(model, l_s, r_s, l_m, r_m):
    # model.load_state_dict(torch.load(file_path, weights_only=True))
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    def sample_output(input) :
        #return(input[0], input[1], input[2], input[3])
        input_tensor = torch.from_numpy(input.reshape(1,4)).float().to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        return output_tensor.cpu().detach().numpy().reshape(-1)

    # print(l_s, r_s, l_m, r_m)
    s = sample_output(np.array([l_s, r_s, l_m, r_m]))
    # print(s)
    return s[2], s[3] # predicted values of (m_l, m_r)


    