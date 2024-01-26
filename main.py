from models import AutoEncoder
from exps import continuous_beam, shear_type_structure, base_isolated_shear_type_structure
import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
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
    # base_isolated_shear_type_structure.compute_response(1)
    sol1 = shear_type_structure.analytical_validation()
    sol2 = base_isolated_shear_type_structure.analytical_validation()
    plt.plot(sol1['time'], sol1['disp'][1, :])
    plt.plot(sol2['time'], sol2['disp'][1, :] + sol2['disp'][0, :])
    plt.show()


    # with open('./dataset/shear_type_structure/solution001.pkl', 'rb') as f:
    #     solution = pickle.load(f)
    # time = solution['time']
    # disp_sts = solution['disp']
    # acc_sts = solution['acc']
    # acc_g_sts = solution['acc_g']
    # plt.plot(time, disp_sts[0, :])

    # with open('./dataset/base_isolated_structure/solution001.pkl', 'rb') as f:
    #     solution = pickle.load(f)
    # time = solution['time']
    # disp = solution['disp']
    # acc = solution['acc']
    # acc_g = solution['acc_g']
    # z = solution['z']
    # plt.plot(time, disp[0, :], '--')

    # plt.xlabel('Time (s)')
    # plt.ylabel('Base displacement (m)')
    # plt.show()


    # plt.plot(time, acc_g_sts)
    # plt.plot(time, acc_g, '--')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Base acceleration (m/s^2)')
    # plt.show()

# print(np.sum(abs(disp_sts[0, :]-disp[0, :])))
# plt.plot(time, disp_sts[0, :]-disp[0, :])
# plt.show()
