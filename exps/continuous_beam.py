import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
from models import Rnn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def random_vibration():
    psd = BandPassPSD(a_v=2.25, f_1=10.0, f_2=410.0)
    force = PSDExcitationGenerator(psd, tmax=10, fmax=500)
    force = force()
    cb = ContinuousBeam01(t_eval=np.linspace(0, 10 - 1 / 1000, 10000), f_t=[force])
    full_data = cb.response(type="full", method="Radau")
    solution = {}
    solution["displacement"] = full_data["displacement"].T
    solution["acceleration"] = full_data["acceleration"].T
    solution["velocity"] = full_data["velocity"].T
    solution["force"] = cb.f_mtx().T
    solution["time"] = cb.t_eval.reshape(-1, 1)
    file_name = f"./dataset/csb/solution" + format(0, "03") + ".pkl"
    with open(file_name, "wb") as f:
        pickle.dump(solution, f)
    print("File " + file_name + " saved.")


def plot_solution():
    with open("./dataset/csb/solution000.pkl", "rb") as f:
        solution = pickle.load(f)
    plt.plot(solution["time"], solution["displacement"][:, 0])
    plt.plot(solution["time"], solution["displacement"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["force"][:, 0])
    plt.show()


def training_test_data(acc_sensor, percent_train):
    filename = "./dataset/csb/solution000.pkl"
    with open(filename, "rb") as f:
        solution = pickle.load(f)
    total_time_step = solution["time"].shape[0]
    train_time_step = int(total_time_step * percent_train)
    state_train = torch.tensor(
        np.hstack(
            solution["displacement"][:train_time_step, :],
            solution["velocity"][:train_time_step, :],
        )
    ).to(device)
    state_test = torch.tensor(
        np.hstack(
            solution["displacement"][train_time_step:, :],
            solution["velocity"][train_time_step:, :],
        )
    ).to(device)
    acc_train = torch.tensor(solution["acceleration"][:train_time_step, acc_sensor]).to(
        device
    )
    acc_test = torch.tensor(solution["acceleration"][train_time_step:, acc_sensor]).to(
        device
    )
    return state_train, acc_train, state_test, acc_test


def _rnn(acc_sensor, percent_train, epochs, lr, weight_decay=0.0):
    state_train, acc_train, state_test, acc_test = training_test_data(
        acc_sensor, percent_train
    )
    train_set = {"X": acc_train, "Y": state_train}
    test_set = {"X": acc_test, "Y": state_test}

    rnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=512,
        num_layers=1,
        output_size=258,
        bidirectional=False,
    )
    train_h0 = torch.zeros(1, 1, 512).to(device)
    train_h0 = torch.zeros(1, 1, 512).to(device)
    model_save_path = f"./dataset/csb/rnn.pth"
    loss_save_path = f"./dataset/csb/rnn.pkl"
    rnn.train(state_train, acc_train)
    train_loss_list, test_loss_list = rnn.train_RNN(
        train_set,
        test_set,
        train_h0,
        train_h0,
        epochs,
        lr,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list
