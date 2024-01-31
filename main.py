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
    pass
    # continuous_beam.compute_response(excitation_pattern=1)
    # continuous_beam.compute_response(excitation_pattern=2)

    # continuous_beam.ae_num_hid_neuron()
    # continuous_beam.ae_train_data_size()

    # continuous_beam.ae_disp_velo()
    # continuous_beam.model_eval()

    # _ = shear_type_structure.compute_response(50)
    # _ = base_isolated_shear_type_structure.compute_response(50)

    shear_type_structure.build_rnn()

    # plt.plot(sol1["time"], sol1["disp"][10, :], label="Shear type structure")
    # plt.plot(
    #     sol2["time"], sol2["disp"][0, :] + sol1["disp"][10, :], label="Base isolated shear type structure"
    # )
    # plt.show()
