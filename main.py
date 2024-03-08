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
random.seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    # base_isolated_shear_type_structure.ambient_response()
    shear_type_structure.modal_analysis()
    shear_type_structure.model_updating(num_modes=5)
    # shear_type_structure.damping_ratio()
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




