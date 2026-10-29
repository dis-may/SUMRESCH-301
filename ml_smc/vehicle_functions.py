from dsms_predictor_nn import dSMSPredictorNN, new_dSMSPredictorNN, sm_to_next_motor_nn
from pylab import *
import torch

from pathlib import Path

def get_model(vehicle_type, folder: str):
    """
    Takes in a vehicle type
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")

    nn_type = None
    if folder == "ml_models_sm_motor":
        nn_type = sm_to_next_motor_nn

    model = nn_type().to(device)
    base_path = Path(__file__).parent.resolve() 
    # Construct a path to a data file relative to that directory
    file_path = base_path / folder / f"{vehicle_type.name.lower()}_model.pth"
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    return model


def get_frankenstein_model(combination=""):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")

    model = sm_to_next_motor_nn().to(device)
    base_path = Path(__file__).parent.resolve() 
    # Construct a path to a data file relative to that directory
    file_path = base_path / "frankenstein" / f"frankenstein_{combination}_model.pth"
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    return model

def get_motor_values(model, inputs):
    # model.load_state_dict(torch.load(file_path, weights_only=True))
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    def sample_output(input) :
        #return(input[0], input[1], input[2], input[3])
        input_tensor = torch.from_numpy(input.reshape(1,len(inputs))).float().to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        return output_tensor.cpu().detach().numpy().reshape(-1)

    # print(l_s, r_s, l_m, r_m)
    s = sample_output(np.array(inputs))
    # print(s)
    return s[0], s[1] # predicted values of (m_l, m_r)


def get_sm_values(model, l_s, r_s, l_m, r_m):
    """
    Returns predicted SM values (for left/right sensors and left/right motors)
    """

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
    return s[0], s[1], s[2], s[3] # predicted values of (s_l, s_r, m_l, m_r)

    