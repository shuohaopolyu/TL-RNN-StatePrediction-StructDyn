from exps import (
    shear_type_structure,
    base_isolated_shear_type_structure,
)
import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from figures import (
    system_identification,
    rnn_training,
    artifical_acc,
    strong_ground_motion,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    # base_isolated_shear_type_structure.ambient_response()
    # shear_type_structure.modal_analysis()
    # shear_type_structure.model_updating(num_modes=5)
    # shear_type_structure.seismic_response(num=100)
    # shear_type_structure.build_birnn()
    # shear_type_structure.build_rnn()
    # shear_type_structure.tune_dkf_params()
    # shear_type_structure.dkf()
    # base_isolated_shear_type_structure.seismic_response()
    # shear_type_structure.birnn_seismic_pred()
    # shear_type_structure.rnn_seismic_pred()
    shear_type_structure.tr_birnn()
    # shear_type_structure.tr_rnn()

    # figure plot
    # system_identification.base_loads()
    # system_identification.singular_values()
    # system_identification.mode_shape()
    # system_identification.natural_frequency()
    # system_identification.params()
    # artifical_acc.cwt_acc_g()
    # rnn_training.loss_curve()
    # rnn_training.disp_pred()
    # rnn_training.velo_pred()
    # rnn_training.performance_evaluation()
    # strong_ground_motion.plot_ground_motion()
