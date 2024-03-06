from models import AutoEncoder
from exps import (
    continuous_beam,
    shear_type_structure,
    base_isolated_shear_type_structure,
)
import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from figures import system_identification
# random.seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    # continuous_beam.compute_response(excitation_pattern=1)
    # continuous_beam.compute_response(excitation_pattern=2)
    # continuous_beam.ae_num_hid_neuron()
    # continuous_beam.ae_train_data_size()
    # continuous_beam.ae_disp_velo()
    # continuous_beam.model_eval()

    # base_isolated_shear_type_structure.ambient_response(save_path="./dataset/bists/")
    # system identification and model updating
    # shear_type_structure.plot_response()
    # base_isolated_shear_type_structure.seismic_response_sample()
    # _ = shear_type_structure.compute_response(50)
    # _ = base_isolated_shear_type_structure.compute_response(50)
    # shear_type_structure.plot_response(1)
    # shear_type_structure.build_rnn()
    # base_isolated_shear_type_structure.lf_rnn_prediction(dof=0)
    # base_isolated_shear_type_structure.build_tr_rnn()

    # figure plot
    # system_identification.acceleration_measurement()
    # system_identification.psd_acc()
    system_identification.mode_shape()
    # system_identification.ms_acc(0.3, 0.8)
    # system_identification.ms_acc(1.3, 1.8)
    # system_identification.ms_acc(2.2, 2.7)
    # system_identification.ms_acc(3.0, 3.6)
    # # system_identification.ms_acc(3.5, 4.1)
    # # system_identification.ms_acc(4.2, 4.7)
    # plt.show()



