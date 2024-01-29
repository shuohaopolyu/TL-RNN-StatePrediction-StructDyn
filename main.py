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

    sol1 = shear_type_structure.compute_response(1)
    sol2 = base_isolated_shear_type_structure.compute_response(1)
    sol3 = shear_type_structure.compute_response(1, method="RK45")
    sol4 = shear_type_structure.compute_response(1, method="RK23") 
    plt.plot(sol1["time"], sol1["acc"][0, :])
    plt.plot(sol2["time"], sol2["acc"][0, :])
    plt.plot(sol3["time"], sol3["acc"][0, :])
    plt.plot(sol4["time"], sol4["acc"][0, :])
    plt.show()

