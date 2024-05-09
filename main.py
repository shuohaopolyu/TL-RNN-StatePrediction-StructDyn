from exps import (
    shear_type_structure,
    base_isolated_shear_type_structure,
    continuous_beam,
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
    tr_rnn_training,
    csb_rnn,
)
from utils import (
    waveform_generator_1,
    waveform_generator_2,
    process_fbg_data,
    process_dewe_data,
    combine_data,
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
    # shear_type_structure.build_dkf()
    # shear_type_structure.build_akf()
    # base_isolated_shear_type_structure.seismic_response()
    # shear_type_structure.birnn_seismic_pred()
    # shear_type_structure.rnn_seismic_pred()
    # shear_type_structure.dkf_seismic_pred()
    # shear_type_structure.integr_dkf_seismic_pred()
    # shear_type_structure.akf_seismic_pred()
    # shear_type_structure.integr_akf_seismic_pred()
    # shear_type_structure.tr_birnn()
    # shear_type_structure.tr_rnn()

    # continuous_beam.ema()
    # continuous_beam.model_updating()
    # continuous_beam.random_vibration(num=15)
    # continuous_beam.plot_solution()
    continuous_beam.build_rnn()
    # continuous_beam.test_rnn()
    # continuous_beam.rnn_pred()
    continuous_beam.tr_rnn()

    # figure plot
    # system_identification.base_loads()
    # system_identification.singular_values()
    # system_identification.model_updating()
    # artifical_acc.cwt_acc_g()
    # rnn_training.loss_curve()
    # rnn_training.state_pred()
    # rnn_training.performance_evaluation()
    # strong_ground_motion.plot_ground_motion()
    # tr_rnn_training.velo_pred()
    # tr_rnn_training.disp_pred()
    # tr_rnn_training.loss_curve()
    # tr_rnn_training.performance_evaluation()
    # csb_rnn.model_updating()

    # experiment utils
    # waveform_generator_1()
    # waveform_generator_2()
    # combine_data()
