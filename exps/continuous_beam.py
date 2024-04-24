import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
from models import Rnn02
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def random_vibration(num=100):
    for i in range(num):
        print(f"Generating solution {i}...")
        start_time = time.time()
        psd = BandPassPSD(a_v=2.25, f_1=10.0, f_2=410.0)
        force = PSDExcitationGenerator(psd, tmax=10, fmax=2000)
        print("Force" + " generated.")
        force = force()
        sampling_freq = 10000
        samping_period = 10.0
        cb = ContinuousBeam01(
            t_eval=np.linspace(
                0,
                samping_period,
                int(sampling_freq * samping_period) + 1,
            ),
            f_t=[force],
        )
        full_data = cb.run()
        solution = {}
        solution["displacement"] = full_data["displacement"].T
        solution["acceleration"] = full_data["acceleration"].T
        solution["velocity"] = full_data["velocity"].T
        solution["force"] = full_data["force"].T
        solution["time"] = full_data["time"]
        file_name = f"./dataset/csb/solution" + format(i, "03") + ".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} s")


def plot_solution():
    with open("./dataset/csb/solution000.pkl", "rb") as f:
        solution = pickle.load(f)
    plt.plot(solution["time"], solution["displacement"][:, 0] * 1000)
    plt.plot(solution["time"], solution["displacement"][:, 36] * 1000)
    plt.show()
    plt.plot(solution["time"], solution["acceleration"][:, 22] / 100)
    plt.plot(solution["time"], solution["acceleration"][:, 44] / 100)
    plt.show()
    plt.plot(solution["time"], solution["velocity"][:, 3])
    plt.plot(solution["time"], solution["velocity"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["force"][:])
    plt.show()


def training_test_data(acc_sensor, num_train_files, num_test_files):
    for i in range(num_train_files):
        filename = f"./dataset/csb/solution" + format(i, "03") + ".pkl"
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        if i == 0:
            total_time_step = solution["time"].shape[0]
            state_train = torch.zeros(num_train_files, total_time_step, 256).to(device)
            acc_train = torch.zeros(
                num_train_files, total_time_step, len(acc_sensor)
            ).to(device)
        state_train[i, :, :] = torch.tensor(
            np.hstack(
                (
                    solution["displacement"] * 1000,
                    solution["velocity"] * 10,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_train[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] / 100, dtype=torch.float32
        ).to(device)
    for i in range(num_test_files):
        filename = (
            f"./dataset/csb/solution" + format(i + num_train_files, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        if i == 0:
            total_time_step = solution["time"].shape[0]
            state_test = torch.zeros(num_test_files, total_time_step, 256).to(device)
            acc_test = torch.zeros(num_test_files, total_time_step, len(acc_sensor)).to(
                device
            )
        state_test[i, :, :] = torch.tensor(
            np.hstack(
                (
                    solution["displacement"] * 1000,
                    solution["velocity"] * 10,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_test[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] / 100, dtype=torch.float32
        ).to(device)
    return state_train, acc_train, state_test, acc_test


def _rnn(acc_sensor, num_train_files, num_test_files, epochs, lr, weight_decay=0.0):
    state_train, acc_train, state_test, acc_test = training_test_data(
        acc_sensor, num_train_files, num_test_files
    )
    train_set = {"X": acc_train, "Y": state_train}
    test_set = {"X": acc_test, "Y": state_test}
    print(f"Train set: {state_train.shape}, {acc_train.shape}")
    print(f"Test set: {state_test.shape}, {acc_test.shape}")

    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    train_h0 = torch.zeros(1, num_train_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    test_h0 = torch.zeros(1, num_test_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    model_save_path = f"./dataset/csb/rnn.pth"
    loss_save_path = f"./dataset/csb/rnn.pkl"
    train_loss_list, test_loss_list = rnn.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs,
        lr,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list


def build_rnn():
    acc_sensor = [24, 44, 64, 100]
    epochs = 20000
    lr = 1e-4
    train_loss_list, test_loss_list = _rnn(acc_sensor, 20, 10, epochs, lr)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()
