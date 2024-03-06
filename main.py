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

# random.seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    # continuous_beam.compute_response(excitation_pattern=1)
    # continuous_beam.compute_response(excitation_pattern=2)
    # continuous_beam.ae_num_hid_neuron()
    # continuous_beam.ae_train_data_size()
    # continuous_beam.ae_disp_velo()
    # continuous_beam.model_eval()

    # base_isolated_shear_type_structure.ambient_response()
    # system identification and model updating
    shear_type_structure.plot_response()
    # base_isolated_shear_type_structure.seismic_response_sample()
    # _ = shear_type_structure.compute_response(50)
    # _ = base_isolated_shear_type_structure.compute_response(50)
    # shear_type_structure.plot_response(1)
    # shear_type_structure.build_rnn()
    # base_isolated_shear_type_structure.lf_rnn_prediction(dof=0)
    # base_isolated_shear_type_structure.build_tr_rnn()


