from models import AutoEncoder
from exps import continuous_beam
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == '__main__':
    
    # continuous_beam.compute_response(excitation_pattern=1)
    # continuous_beam.compute_response(excitation_pattern=2)
    continuous_beam.ae_num_hid_neuron()
    # continuous_beam.ae_train_data_size()