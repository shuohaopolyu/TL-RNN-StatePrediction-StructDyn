from systems import ShearTypeStructure
import numpy as np
import pickle
from utils import compute_metrics
from models import Rnn, Lstm
import torch
import matplotlib.pyplot as plt


def _read_smc_file(filename):
    with open(filename) as f:
        content = f.readlines()
    data = []
    counter = 0
    for x in content:
        if counter > 39:
            s = x
            numbers = [s[i : i + 10] for i in range(0, len(s), 10)]
            number = []
            for i in range(len(numbers) - 1):
                number.append(float(numbers[i]))
            data.append(number)
        counter += 1
    acc = np.array(data)
    time = np.linspace(0, 0.02 * (len(acc[:, 0]) - 1), len(acc[:, 0]))
    return time, acc[:, 0] * 1e-2


def compute_response(num=1, method="Radau"):
    acc_file_root_name = "./excitations/SMSIM/m7.0r10.0_00"

    acc_file_list = [
        acc_file_root_name + format(i, "03") + ".smc" for i in range(1, num + 1)
    ]

    for acc_file_i in acc_file_list:
        time, acc = _read_smc_file(acc_file_i)
        time = time[1000:6000]
        acc = acc[1000:6000]
        time = time - time[0]
        mass_vec = 3e5 * np.ones(13)
        stiff_vec = 9e6 * np.ones(13)
        damp_vec = 3e5 * np.ones(13)
        mass_vec[0] = 6e5
        damp_vec[0] = 5e5
        stiff_vec[0] = 3e6

        parametric_sts = ShearTypeStructure(
            mass_vec=mass_vec,
            stiff_vec=stiff_vec,
            damp_vec=damp_vec,
            t=time,
            acc_g=acc,
        )

        acc, velo, disp = parametric_sts.run(method)

        solution = {
            "acc_g": parametric_sts.acc_g,
            "time": time,
            "disp": disp,
            "velo": velo,
            "acc": acc,
        }

        file_name = (
            "./dataset/shear_type_structure/solution"
            + format(acc_file_list.index(acc_file_i) + 1, "03")
            + ".pkl"
        )
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")
        _ = parametric_sts.print_damping_ratio(10)
        _ = parametric_sts.print_natural_frequency(10)
    return solution

def plot_response(which=1):
    filename = (
            "./dataset/shear_type_structure/solution" + format(which, "03") + ".pkl"
        )
    with open(filename, "rb") as f:
        solution = pickle.load(f)
    
    time = solution["time"]
    acc = solution["acc"]
    disp = solution["disp"]
    velo = solution["velo"]
    acc_g = solution["acc_g"]
    plt.plot(time, disp[0, :].T, label="Ground truth")
    # plt.show()


def analytical_validation(method="Radau"):
    time = np.linspace(0, 10, 10000)
    acc = np.sin(2 * np.pi * 1 * time)
    mass_vec = 2 * np.ones(3)
    stiff_vec = 10 * np.ones(3)
    damp_vec = 0.1 * np.ones(3)
    mass_vec[0] = 1
    sts = ShearTypeStructure(
        mass_vec=mass_vec,
        stiff_vec=stiff_vec,
        damp_vec=damp_vec,
        t=time,
        acc_g=acc,
    )
    print(sts.mass_mtx)

    acc, velo, disp = sts.run(method)

    solution = {
        "acc_g": sts.acc_g,
        "time": time,
        "disp": disp,
        "velo": velo,
        "acc": acc,
    }
    _ = sts.print_damping_ratio(3)
    return solution


def _rnn(
    acc_sensor,
    data_compression_ratio=10,
    num_training_files=10,
    epochs=10000,
    lr=0.0001,
    weight_decay=0.0,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    num_test_files = 50 - num_training_files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disp = []
    acc = []
    disp_test = []
    acc_test = []

    for i in range(num_training_files):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp.append(solution["disp"][:, ::data_compression_ratio].T)
        acc.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)

    for i in range(num_training_files, 50):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test.append(solution["disp"][:, ::data_compression_ratio].T)
        acc_test.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)

    disp = np.array(disp)
    disp = torch.tensor(disp, dtype=torch.float32).to(device)
    disp_test = np.array(disp_test)
    disp_test = torch.tensor(disp_test, dtype=torch.float32).to(device)
    acc = np.array(acc)
    acc = torch.tensor(acc, dtype=torch.float32).to(device)
    acc_test = np.array(acc_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    train_set = {"X": acc, "Y": disp}
    test_set = {"X": acc_test, "Y": disp_test}

    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    train_h0 = torch.zeros(2, num_training_files, 10).to(device)
    test_h0 = torch.zeros(2, num_test_files, 10).to(device)
    model_save_path = "./dataset/shear_type_structure/rnn_disp.pth"
    loss_save_path = "./dataset/shear_type_structure/rnn_disp_loss.pkl"
    train_loss_list, test_loss_list = RNN_model_disp.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs=epochs,
        learning_rate=lr,
        model_save_path=model_save_path,
        loss_save_path=loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )

    return train_loss_list, test_loss_list


def build_rnn():
    dr = 10
    ntf = 40
    acc_sensor = [0, 4, 7, 11]
    _, _ = _rnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        epochs=200000,
        lr=0.0001,
        weight_decay=0.0,
    )
    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/shear_type_structure/rnn_disp.pth", "rb") as f:
        RNN_model_disp.load_state_dict(torch.load(f))

    RNN_model_disp.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 50 - ntf
    test_h0 = torch.zeros(2, num_test_files, 10).to(device)
    disp_test = []
    acc_test = []
    for i in range(ntf, 50):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test.append(solution["disp"][:, ::dr].T)
        acc_test.append(solution["acc"][acc_sensor, ::dr].T)
    disp_test = np.array(disp_test)
    acc_test = np.array(acc_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        disp_pred,_ = RNN_model_disp(acc_test, test_h0)
    disp_pred = disp_pred.cpu().numpy()
    disp_test = disp_test[:, :, 11]
    disp_pred = disp_pred[:, :, 11]
    plt.plot(disp_test[0, :], label="Ground truth")
    plt.plot(disp_pred[0, :], label="Prediction")
    plt.legend()
    plt.show()
