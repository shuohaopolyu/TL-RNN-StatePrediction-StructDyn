import numpy as np
import pickle
from systems import ContinuousBridge
from excitations import PSDExcitationGenerator, RollOffPSD, FlatNoisePSD
import torch
from models import AutoEncoder
import os
from utils import compute_metrics


def compute_response(excitation_pattern=1):
    # Define the force
    if excitation_pattern == 1:
        psd = FlatNoisePSD(a_v=1e8)
    elif excitation_pattern == 2:
        psd = RollOffPSD(a_v=1e12, omega_r=50.0, omega_c=80.0)
    force = PSDExcitationGenerator(psd=psd, tmax=100, fmax=50)
    force_fun_1, force_fun_2, force_fun_3, force_fun_4 = (
        force(),
        force(),
        force(),
        force(),
    )

    # Define the system
    cb = ContinuousBridge(
        rho=1e4,
        E=6e9,
        A=1,
        I=1 / 12,
        L=0.5,
        damp_params=(0, 5, 0.03),
        f_dof=[13, 30, 44, 69],
        resp_dof="full",
        t_eval=np.arange(0, 100, 1 / 100),
        f_t=[force_fun_1, force_fun_2, force_fun_3, force_fun_4],
        init_cond=np.zeros(172),
    )

    # generate the data
    full_data = cb.response(type="full", method="Radau")
    full_data["displacement"] = full_data["displacement"].T
    full_data["velocity"] = full_data["velocity"].T
    full_data["acceleration"] = full_data["acceleration"].T
    full_data["force"] = cb.f_mtx().T
    full_data["time"] = cb.t_eval.reshape(-1, 1)

    # save the data
    with open(
        f"./dataset/full_response_excitation_pattern_{excitation_pattern}.pkl", "wb"
    ) as f:
        pickle.dump(full_data, f)


def ae_num_hid_neuron():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data_1 = pickle.load(f)
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data_2 = pickle.load(f)
    disp_1 = torch.tensor(full_data_1["displacement"], dtype=torch.float32).to(device)
    velo_1 = torch.tensor(full_data_1["velocity"], dtype=torch.float32).to(device)
    disp_2 = torch.tensor(full_data_2["displacement"], dtype=torch.float32).to(device)
    velo_2 = torch.tensor(full_data_2["velocity"], dtype=torch.float32).to(device)

    train_set_disp_1 = disp_1[:2000, :]
    test_set_disp_1 = disp_1[2000:10000, :]
    test_set_disp_2 = disp_2
    train_set_velo_1 = velo_1[:2000, :]
    test_set_velo_1 = velo_1[2000:10000, :]
    test_set_velo_2 = velo_2

    error_1 = []
    error_2 = []
    error_3 = []
    error_4 = []
    for i in range(1, 15):
        ae_disp = AutoEncoder([86, 40, 40, i])
        ae_velo = AutoEncoder([86, 40, 40, i])
        ae_disp.train_AE(
            train_set_disp_1,
            test_set_disp_1,
            epochs=10000,
            learning_rate=1e-3,
            model_save_path="./dataset/ae_disptest_" + str(i) + ".pth",
            loss_save_path=None,
        )
        ae_velo.train_AE(
            train_set_velo_1,
            test_set_velo_1,
            epochs=10000,
            learning_rate=1e-3,
            model_save_path="./dataset/ae_velotest_" + str(i) + ".pth",
            loss_save_path=None,
        )
        ae_disp.eval()
        ae_velo.eval()
        with torch.no_grad():
            disp_pred_1 = ae_disp(test_set_disp_1)
            disp_pred_2 = ae_disp(test_set_disp_2)
            velo_pred_1 = ae_velo(test_set_velo_1)
            velo_pred_2 = ae_velo(test_set_velo_2)
            # Compute and append all metrics at once
        error_1.append(compute_metrics(disp_pred_1, test_set_disp_1))
        error_2.append(compute_metrics(disp_pred_2, test_set_disp_2))
        error_3.append(compute_metrics(velo_pred_1, test_set_velo_1))
        error_4.append(compute_metrics(velo_pred_2, test_set_velo_2))

    error_1 = np.array(error_1)
    error_2 = np.array(error_2)
    error_3 = np.array(error_3)
    error_4 = np.array(error_4)

    with open("./dataset/ae_num_hid_neuron.pkl", "wb") as f:
        pickle.dump(
            {
                "error_1": error_1,
                "error_2": error_2,
                "error_3": error_3,
                "error_4": error_4,
            },
            f,
        )

def ae_train_data_size():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data_1 = pickle.load(f)
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data_2 = pickle.load(f)
    disp_1 = torch.tensor(full_data_1["displacement"], dtype=torch.float32).to(device)
    velo_1 = torch.tensor(full_data_1["velocity"], dtype=torch.float32).to(device)
    disp_2 = torch.tensor(full_data_2["displacement"], dtype=torch.float32).to(device)
    velo_2 = torch.tensor(full_data_2["velocity"], dtype=torch.float32).to(device)
    num_data = [100, 200, 500, 1000, 2000, 4000, 8000]

    for i in num_data:
        train_set_disp_1 = disp_1[:i, :]
        test_set_disp_1 = disp_1[i:10000, :]
        test_set_disp_2 = disp_2
        train_set_velo_1 = velo_1[:i, :]
        test_set_velo_1 = velo_1[i:10000, :]
        test_set_velo_2 = velo_2

        ae_disp = AutoEncoder([86, 40, 40, 5])
        ae_velo = AutoEncoder([86, 40, 40, 5])
        ae_disp.train_AE(
            train_set_disp_1,
            test_set_disp_1,
            epochs=10000,
            learning_rate=1e-3,
            model_save_path="./dataset/ae_disptest_" + str(i) + ".pth",
            loss_save_path=None,
        )
        ae_velo.train_AE(
            train_set_velo_1,
            test_set_velo_1,
            epochs=10000,
            learning_rate=1e-3,
            model_save_path="./dataset/ae_velotest_" + str(i) + ".pth",
            loss_save_path=None,
        )


