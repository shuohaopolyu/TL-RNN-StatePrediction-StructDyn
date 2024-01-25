from models import AutoEncoder
from exps import continuous_beam, shear_type_structure, base_isolated_shear_type_structure
import os
import random
import pickle
import matplotlib.pyplot as plt
# random.seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == '__main__':
    
    # continuous_beam.compute_response(excitation_pattern=1)
    # continuous_beam.compute_response(excitation_pattern=2)

    # continuous_beam.ae_num_hid_neuron()
    # continuous_beam.ae_train_data_size()

    # continuous_beam.ae_disp_velo()
    # continuous_beam.model_eval()

    # shear_type_structure.compute_response(1)
    # shear_type_structure.plot_response()

    # base_isolated_shear_type_structure.compute_response(1)
    # base_isolated_shear_type_structure.plot_response()

    with open('./dataset/shear_type_structure/solution001.pkl', 'rb') as f:
        solution = pickle.load(f)
    time = solution['time']
    disp_sts = solution['disp']
    # plt.plot(time, disp_sts[0, :])
    plt.plot(time, disp_sts[11, :])
    plt.plot(time, disp_sts[12, :])

    with open('./dataset/base_isolated_structure/solution001.pkl', 'rb') as f:
        solution = pickle.load(f)
    time = solution['time']
    disp = solution['disp']
    plt.plot(time, disp[0, :])
    plt.plot(time, disp[0, :]+disp[11, :])

    # plt.plot(time, disp[1, :]+disp[0, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Base displacement (m)')
    plt.show()