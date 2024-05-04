import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
from models import Rnn02
import torch
import time
import scipy.io
from scipy import signal
import sdypy as sd
from scipy.optimize import minimize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def _measured_strain(filename=f"./dataset/csb/exp_1.mat"):
    k = 0.78
    # measured strain data for training
    exp_data = scipy.io.loadmat(filename)
    fbg1_ini = exp_data["fbg1_ini"]
    fbg2_ini = exp_data["fbg2_ini"]
    fbg3_ini = exp_data["fbg3_ini"]
    fbg4_ini = exp_data["fbg4_ini"]
    strain1 = (exp_data["fbg1"][::5] - fbg1_ini) / (fbg1_ini * k)
    strain2 = (exp_data["fbg2"][::5] - fbg2_ini) / (fbg2_ini * k)
    strain3 = (exp_data["fbg3"][::5] - fbg3_ini) / (fbg3_ini * k)
    strain4 = (exp_data["fbg4"][::5] - fbg4_ini) / (fbg4_ini * k)
    strain = np.hstack((strain1, strain2, strain3, strain4))
    strain = torch.tensor(strain, dtype=torch.float32).to(device) * 1e6
    return strain


def _measured_acc(filename=f"./dataset/csb/exp_1.mat", acc_scale=0.01):
    exp_data = scipy.io.loadmat(filename)
    # print(exp_data.keys())
    acc1 = exp_data["acc1"][::5] * 9.8 * acc_scale
    acc2 = exp_data["acc2"][::5] * 9.8 * acc_scale
    acc3 = exp_data["acc3"][::5] * 9.8 * acc_scale
    acc4 = exp_data["acc4"][::5] * 9.8 * acc_scale
    acc_tensor = torch.tensor(
        -np.hstack((acc4, acc3, acc2, acc1)), dtype=torch.float32
    ).to(device)
    # acc_tensor = torch.tensor(
    #     np.hstack((acc1, acc2, acc3, acc4)), dtype=torch.float32
    # ).to(device)
    # plt.plot(acc_tensor.detach().cpu().numpy())
    # plt.show()
    return acc_tensor


def _measured_disp(filename=f"./dataset/csb/exp_1.mat", disp_scale=10000):
    # the measured displacement data, unit: mm
    exp_data = scipy.io.loadmat(filename)
    disp = exp_data["disp1"][::5]
    disp = torch.tensor(disp, dtype=torch.float32).to(device) * disp_scale
    return disp


def _measured_force(filename=f"./dataset/csb/exp_1.mat"):
    # the measured force data, unit: N
    exp_data = scipy.io.loadmat(filename)
    force = exp_data["force1"][::5]
    force = torch.tensor(force, dtype=torch.float32).to(device)
    return force


def ema_freq_sweep():
    save_path = "./dataset/csb/ema.pkl"
    data_path = "./dataset/csb/sweep.mat"
    exp_data = scipy.io.loadmat(data_path)
    acc1 = exp_data["Data1_AI_1"][::5, 0].reshape(-1, 1) * 9.8
    acc2 = exp_data["Data1_AI_2"][::5, 0].reshape(-1, 1) * 9.8
    acc3 = exp_data["Data1_AI_3"][::5, 0].reshape(-1, 1) * 9.8
    acc4 = exp_data["Data1_AI_4"][::5, 0].reshape(-1, 1) * 9.8
    acc_mtx = -np.hstack((acc4, acc3, acc2, acc1))
    acc_mtx = np.array([acc_mtx.T], dtype=np.float32)
    print(acc_mtx.shape)
    force = exp_data["Data1_AI_5"][::5, 0].reshape(-1, 1)
    force_mtx = np.array([force.T], dtype=np.float32)
    frf_obj = sd.FRF.FRF(
        1000,
        force_mtx,
        acc_mtx,
        window="hann",
        fft_len=1000,
        nperseg=1000,
    )
    frf = frf_obj.get_FRF(type="default", form="accelerance")
    frf = frf.squeeze()
    plt.plot(np.abs(frf.T))
    plt.legend(["acc1", "acc2", "acc3", "acc4"])
    plt.yscale("log")
    plt.show()
    frf = frf[:, 10:200].T
    with open(save_path, "wb") as f:
        pickle.dump(frf, f)
    print(f"EMA data saved to {save_path}")


def modal_analysis_ema(acc_scale=0.01):
    save_path = "./dataset/csb/ema.pkl"
    data_path = "./dataset/csb/exp_1.mat"
    acc_tensor = _measured_acc(filename=data_path)
    acc_mtx1 = acc_tensor.detach().cpu().numpy().T
    force_tensor = _measured_force(filename=data_path)
    force_mtx1 = force_tensor.detach().cpu().numpy().T
    data_path = "./dataset/csb/exp_2.mat"
    acc_tensor = _measured_acc(filename=data_path)
    acc_mtx2 = acc_tensor.detach().cpu().numpy().T
    force_tensor = _measured_force(filename=data_path)
    force_mtx2 = force_tensor.detach().cpu().numpy().T
    data_path = "./dataset/csb/exp_3.mat"
    acc_tensor = _measured_acc(filename=data_path)
    acc_mtx3 = acc_tensor.detach().cpu().numpy().T
    force_tensor = _measured_force(filename=data_path)
    force_mtx3 = force_tensor.detach().cpu().numpy().T
    acc_mtx = np.array(
        [acc_mtx1 / acc_scale, acc_mtx2 / acc_scale, acc_mtx3 / acc_scale]
    )
    force_mtx = np.array([force_mtx1, force_mtx2, force_mtx3])
    frf_obj = sd.FRF.FRF(
        1000,
        force_mtx,
        acc_mtx,
        window="hann",
        nperseg=1000,
    )
    frf = frf_obj.get_FRF(type="default", form="accelerance")
    # frf = frf_obj.get_Hv()
    frf = frf.squeeze()
    frf = frf[:, ::4]
    freq = np.linspace(0, 500, 501)
    plt.plot(freq, np.abs(frf.T))
    plt.legend(["1", "2", "3", "4"])
    plt.yscale("log")
    plt.xlim([10, 200])
    plt.show()
    frf = frf[:, 10:200].T
    with open(save_path, "wb") as f:
        pickle.dump(frf, f)
    print(f"EMA data saved to {save_path}")


def modal_properties(i=5):
    cb = ContinuousBeam01(
        t_eval=np.linspace(0, 10, 10001),
        f_t=[0],
    )
    u, v = cb.freqs_modes()
    v = v[::2, :]
    frf_mtx = cb.frf()
    print(frf_mtx.shape)


def compute_frf(log_k_theta, log_k_theta_1, log_k_theta_2, log_k_t, E_factor, damp):
    k_theta = 10**log_k_theta
    k_theta_1 = 10**log_k_theta_1
    k_theta_2 = 10**log_k_theta_2
    k_t = 10**log_k_t
    material_properties = {
        "elastic_modulus": 70e9 * E_factor,
        "density": 2.7e3,
        "mid_support_rotational_stiffness": k_theta,
        "mid_support_transversal_stiffness": k_t,
        "end_support_rotational_stiffness_1": k_theta_1,
        "end_support_rotational_stiffness_2": k_theta_2,
    }

    cb = ContinuousBeam01(
        t_eval=np.linspace(0, 10, 1001),
        f_t=[1],
        material_properties=material_properties,
        damping_params=(0, 1, damp),
    )
    frf_mtx = cb.frf()
    return frf_mtx


def loss_func(params, frf_data):
    log_k_theta, log_k_theta_1, log_k_theta_2, log_k_t, E_factor, damp = params
    frf_mtx = compute_frf(
        log_k_theta, log_k_theta_1, log_k_theta_2, log_k_t, E_factor, damp
    )
    # magnitude_error = np.sum(
    #     (np.log10(np.abs(frf_mtx)) - np.log10(np.abs(frf_data))) ** 2
    # )
    # phase_error = np.sum((np.angle(frf_mtx) - np.angle(frf_data)) ** 2)
    # loss = magnitude_error + phase_error
    loss = np.sum((np.log10(np.abs(frf_mtx)) - np.log10(np.abs(frf_data))) ** 2)
    # loss = np.sum(np.abs(np.log10(np.abs(frf_mtx)) - np.log10(np.abs(frf_data))))
    return loss


def compare_frf(result, frf_data):
    log_k_theta, log_k_theta_1, log_k_theta_2, log_k_t, E_factor, damp = result.x
    frf_mtx = compute_frf(
        log_k_theta, log_k_theta_1, log_k_theta_2, log_k_t, E_factor, damp
    )
    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    for i in range(4):
        ax[i].plot(np.abs(frf_mtx[:, i]))
        ax[i].plot(np.abs(frf_data[:, i]))

    plt.legend(["Model", "Data"])
    plt.yscale("log")
    plt.show()


def model_updating():
    with open("./dataset/csb/ema.pkl", "rb") as f:
        frf_data = pickle.load(f)
    # Initial guess for parameters
    initial_guess = [3, 7, 7, 4, 1]
    # Minimize the loss function
    result = minimize(
        loss_func,  # function to minimize
        x0=initial_guess,  # initial guess
        args=(frf_data,),  # additional arguments passed to loss_func
        method="L-BFGS-B",  # optimization method
        options={"disp": True},
        bounds=[(0, 6), (7, 8), (7, 8), (3, 6), (0.5, 2)],
    )
    print(result)
    compare_frf(result, frf_data)


def random_vibration(num=30):
    for i in range(num):
        print(f"Generating solution {i}...")
        start_time = time.time()
        psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=410.0)
        force = PSDExcitationGenerator(
            psd, tmax=5, fmax=2000, normalize=True, normalize_factor=5.0
        )
        # force.plot()
        print("Force" + " generated.")
        force = force()
        sampling_freq = 3000
        samping_period = 5.0
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
    print(solution["time"])
    plt.plot(solution["time"], solution["displacement"][:, 0])
    plt.plot(solution["time"], solution["displacement"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["acceleration"][:, 22])
    plt.plot(solution["time"], solution["acceleration"][:, 44])
    plt.show()
    plt.plot(solution["time"], solution["velocity"][:, 3])
    plt.plot(solution["time"], solution["velocity"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["force"][:])
    plt.show()


def training_test_data(
    acc_sensor,
    num_train_files,
    num_test_files,
    disp_scale=10000,
    velo_scale=100,
    acc_scale=0.01,
):
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
                    solution["displacement"] * disp_scale,
                    solution["velocity"] * velo_scale,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_train[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] * acc_scale, dtype=torch.float32
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
                    solution["displacement"] * disp_scale,
                    solution["velocity"] * velo_scale,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_test[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] * acc_scale, dtype=torch.float32
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
    acc_sensor = [24, 44, 64, 98]
    epochs = 20000
    lr = 1e-4
    train_loss_list, test_loss_list = _rnn(acc_sensor, 20, 10, epochs, lr)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()


def rnn_pred(path="./dataset/csb/rnn.pth"):
    acc_sensor = [24, 44, 64, 98]
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    # load the experimental data in .mat format
    acc_tensor = _measured_acc(filename=f"./dataset/csb/exp_4.mat")
    disp_tensor = _measured_disp(filename=f"./dataset/csb/exp_4.mat")
    disp = disp_tensor.detach().cpu().numpy()
    train_h0 = torch.zeros(1, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(acc_tensor, train_h0)
    state_pred = state_pred.detach().cpu().numpy()
    plt.plot(state_pred[:, 34])
    plt.plot(disp)
    plt.show()
    # plt.plot(state_pred[:, 34], disp, "o")
    # plt.show()


def _comp_strain_from_nodal_disp(nodal_disp, loc_fbg):
    # compute strain from nodal displacement
    # nodal_disp: (nt, n_dof), torch tensor
    # loc_fbg: list of float, location of FBG sensors
    num_fbg = len(loc_fbg)
    strain = torch.zeros(nodal_disp.shape[0], num_fbg).to(device)
    y = 0.0025
    L = 0.02
    for i, loc in enumerate(loc_fbg):
        ele_num = int(loc / L)
        dofs = [2 * ele_num, 2 * ele_num + 1, 2 * ele_num + 2, 2 * ele_num + 3]
        disp = nodal_disp[:, dofs] * 1e-4
        x = loc - ele_num * L
        B_mtx = (
            y
            / L
            * torch.tensor(
                [
                    [6 / L - 12 * x / L**2],
                    [4 - 6 * x / L],
                    [-6 / L + 12 * x / L**2],
                    [2 - 6 * x / L],
                ],
                dtype=torch.float32,
            ).to(device)
        )
        strain[:, i] = (disp @ B_mtx).squeeze() * 1e6
    return strain


def tr_training(
    RNN4state,
    acc_tensor,
    measured_strain_train,
    measured_strain_test,
    loc_fbg_train,
    loc_fbg_test,
    lr,
    epochs,
    unfrozen_params,
    save_path,
):
    loss_fun = torch.nn.MSELoss(reduction="mean")
    for j, param in enumerate(RNN4state.parameters()):
        param.requires_grad = False
        if j in unfrozen_params:
            param.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, RNN4state.parameters()), lr=lr
    )
    train_loss_list = []
    test_loss_list = []
    if RNN4state.bidirectional:
        h0 = torch.zeros(2, 1, RNN4state.hidden_size, dtype=torch.float32).to(device)
    else:
        h0 = torch.zeros(1, RNN4state.hidden_size, dtype=torch.float32).to(device)
    for i in range(epochs):
        state_pred, _ = RNN4state(acc_tensor, h0)
        disp_pred = state_pred[:, :128]
        strain_train_pred = _comp_strain_from_nodal_disp(disp_pred, loc_fbg_train)
        optimizer.zero_grad()
        RNN4state.train()
        loss = loss_fun(strain_train_pred, measured_strain_train)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        with torch.no_grad():
            state_pred_test, _ = RNN4state(acc_tensor, h0)
            disp_pred_test = state_pred_test[:, :128]
            strain_test_pred = _comp_strain_from_nodal_disp(
                disp_pred_test, loc_fbg_test
            )
            test_loss = loss_fun(strain_test_pred, measured_strain_test)
            test_loss_list.append(test_loss.item())
        torch.save(RNN4state.state_dict(), save_path)
        if i % 200 == 0:
            print(
                f"Epoch {i}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}"
            )
            plt.plot(
                measured_strain_test[:, 0].detach().cpu().numpy(), label="measured"
            )
            plt.plot(strain_test_pred[:, 0].detach().cpu().numpy(), label="predicted")
            plt.legend()
            plt.show()
            rnn_pred(path=save_path)
    return train_loss_list, test_loss_list


def tr_rnn():
    acc_sensor = [24, 44, 64, 98]
    # transfer learning of recurrent neural networks
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(f"./dataset/csb/rnn.pth"))
    rnn.to(device)
    unfrozen_params = [0, 1]
    lr = 1e-5
    epochs = 2000
    measured_strain = _measured_strain(f"./dataset/csb/exp_3.mat")
    measured_strain_train = measured_strain[:, [0, 2, 3]]
    measured_strain_test = measured_strain[:, [1]]
    loc_fbg_train = [0.3, 0.64, 1.0]
    loc_fbg_test = [0.5]
    acc_tensor = _measured_acc(f"./dataset/csb/exp_3.mat")
    train_loss_list, test_loss_list = tr_training(
        rnn,
        acc_tensor,
        measured_strain_train,
        measured_strain_test,
        loc_fbg_train,
        loc_fbg_test,
        lr,
        epochs,
        unfrozen_params,
        "./dataset/csb/tr_rnn.pth",
    )
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()
