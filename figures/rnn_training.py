import numpy as np
import matplotlib.pyplot as plt
import torch
from models import Rnn
from exps import shear_type_structure
import pickle
from utils import similarity
# set the fonttype to be Arial
plt.rcParams["font.family"] = "Arial"
# set the font size's default value
plt.rcParams.update({"font.size": 10})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def loss_curve():
    loss_save_path = "./dataset/sts/birnn.pkl"
    with open(loss_save_path, "rb") as f:
        birnn = torch.load(f)
    loss_save_path = "./dataset/sts/rnn.pkl"
    with open(loss_save_path, "rb") as f:
        rnn = torch.load(f)
    epoch = np.arange(1, (len(birnn["train_loss_list"])) * 200, 200)
    plt.figure(figsize=(10 * cm, 8 * cm))
    plt.plot(epoch, birnn["train_loss_list"], color="r", linewidth=1.5)
    plt.plot(epoch, birnn["test_loss_list"], color="r", linestyle="--", linewidth=1.5)
    plt.plot(epoch, rnn["train_loss_list"], color="b", linewidth=1.5)
    plt.plot(epoch, rnn["test_loss_list"], color="b", linestyle="--", linewidth=1.5)
    plt.tick_params(which="both", direction="in")
    plt.legend(
        ["BiRNN training", "BiRNN test", "RNN training", "RNN test"],
        fontsize=10,
        facecolor="white",
        edgecolor="black",
    )
    plt.xlim(0, 50000)
    plt.xticks([0, 10000, 20000, 30000, 40000, 50000], ["0", "1", "2", "3", "4", "5"])
    plt.yscale("log")
    plt.xlabel(r"Epoch ($\times 10^4$)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("./figures/loss_curve.svg", bbox_inches="tight")
    plt.show()


def disp_pred():
    file_idx = 2
    dof_idx = 8
    BiRNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    RNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    time = np.arange(0, 40, 1/20)
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    dkf_pred = dkf_pred["disp_pred"]
    RNN4ststate.eval()
    BiRNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 10
    rnn_test_h0 = torch.zeros(1, num_test_files, 30).to(device)
    birnn_test_h0 = torch.zeros(2, num_test_files, 30).to(device)
    _, _, state_test, acc_test = shear_type_structure.training_test_data(
        [0, 1, 2, 3, 4], 1, 90, 100
    )
    with torch.no_grad():
        rnn_state_pred, _ = RNN4ststate(acc_test, rnn_test_h0)
        birnn_state_pred, _ = BiRNN4ststate(acc_test, birnn_test_h0)

    rnn_state_pred = rnn_state_pred.cpu().numpy()
    rnn_state_pred = rnn_state_pred[:, :, dof_idx]
    birnn_state_pred = birnn_state_pred.cpu().numpy()
    birnn_state_pred = birnn_state_pred[:, :, dof_idx]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, dof_idx]
    dkf_state_pred = dkf_pred[:, :, dof_idx]
    dkf_state_pred[file_idx, 1:] = dkf_state_pred[file_idx, :-1]
    dkf_state_pred[file_idx, 0] = 0
    plt.figure(figsize=(20 * cm, 8 * cm))
    plt.plot(time, state_test[file_idx, :]*100, label="Ref.", color="k", linewidth=1.2)
    plt.plot(time,
        birnn_state_pred[file_idx, :]*100,
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    plt.plot(time,
        rnn_state_pred[file_idx, :]*100,
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    plt.plot(time,
        dkf_state_pred[file_idx, :]*100,
        label="DKF pred.",
        linestyle=":",
        color="g",
        linewidth=1.2,
    )
    plt.legend(
        fontsize=10,
        facecolor="white",
        edgecolor="black",
        ncol=4
    )
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (cm)")
    plt.tick_params(which="both", direction="in")
    plt.xlim(0, 40)
    plt.ylim(-0.5*100, 0.5*100)
    plt.text(0.9, 0.125, "9th floor", ha="center", va="center", transform=plt.gca().transAxes)
    plt.savefig("./figures/disp_pred.svg", bbox_inches="tight")
    plt.show()


def velo_pred():
    file_idx = 2
    dof_idx = -1
    BiRNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    RNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    time = np.arange(0, 40, 1/20)
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    RNN4ststate.eval()
    BiRNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 10
    rnn_test_h0 = torch.zeros(1, num_test_files, 30).to(device)
    birnn_test_h0 = torch.zeros(2, num_test_files, 30).to(device)
    _, _, state_test, acc_test = shear_type_structure.training_test_data(
        [0, 1, 2, 3, 4], 1, 90, 100
    )
    with torch.no_grad():
        rnn_state_pred, _ = RNN4ststate(acc_test, rnn_test_h0)
        birnn_state_pred, _ = BiRNN4ststate(acc_test, birnn_test_h0)

    rnn_state_pred = rnn_state_pred.cpu().numpy()
    rnn_state_pred = rnn_state_pred[:, :, dof_idx]
    birnn_state_pred = birnn_state_pred.cpu().numpy()
    birnn_state_pred = birnn_state_pred[:, :, dof_idx]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, dof_idx]
    dkf_pred = dkf_pred["velo_pred"]
    dkf_state_pred = dkf_pred[:, :, dof_idx]
    dkf_state_pred[file_idx, 1:] = dkf_state_pred[file_idx, :-1]
    dkf_state_pred[file_idx, 0] = 0
    plt.figure(figsize=(20 * cm, 8 * cm))
    plt.plot(time, state_test[file_idx, :], label="Ref.", color="k", linewidth=1.2)
    plt.plot(time,
        birnn_state_pred[file_idx, :],
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    plt.plot(time,
        rnn_state_pred[file_idx, :],
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    plt.plot(time,
        dkf_state_pred[file_idx, :],
        label="DKF pred.",
        linestyle=":",
        color="g",
        linewidth=1.2,
    )
    plt.legend(
        fontsize=10,
        facecolor="white",
        edgecolor="black",
        ncol=4
    )
    plt.text(0.9, 0.125, "13th floor", ha="center", va="center", transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.tick_params(which="both", direction="in")
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.savefig("./figures/velo_pred.svg", bbox_inches="tight")
    plt.show()

def performance_evaluation():
    BiRNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    RNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    RNN4ststate.eval()
    BiRNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 10
    rnn_test_h0 = torch.zeros(1, num_test_files, 30).to(device)
    birnn_test_h0 = torch.zeros(2, num_test_files, 30).to(device)
    _, _, state_test, acc_test = shear_type_structure.training_test_data(
        [0, 1, 2, 3, 4], 1, 90, 100
    )
    with torch.no_grad():
        rnn_state_pred, _ = RNN4ststate(acc_test, rnn_test_h0)
        birnn_state_pred, _ = BiRNN4ststate(acc_test, birnn_test_h0)

    rnn_state_pred = rnn_state_pred.cpu().numpy()
    birnn_state_pred = birnn_state_pred.cpu().numpy()
    state_test = state_test.cpu().numpy()
    dkf_pred_disp = dkf_pred["disp_pred"]
    dkf_pred_velo = dkf_pred["velo_pred"]
    dkf_pred_disp[:,1:, :] = dkf_pred_disp[:,:-1, :]
    dkf_pred_disp[:,0, :] = 0
    dkf_pred_velo[:,1:, :] = dkf_pred_velo[:,:-1, :]
    dkf_pred_velo[:,0, :] = 0
    err_mtx_rnn = similarity(rnn_state_pred, state_test)
    err_mtx_birnn = similarity(birnn_state_pred, state_test)
    err_mtx_dkf_disp = similarity(dkf_pred_disp, state_test)
    err_mtx_dkf_velo = similarity(dkf_pred_velo, state_test[:,:,13:])
    mean_err_rnn_disp = np.mean(err_mtx_rnn[:, 0:13])
    mean_err_birnn_disp = np.mean(err_mtx_birnn[:, 0:13])
    mean_err_dkf_disp = np.mean(err_mtx_dkf_disp)
    mean_err_rnn_velo = np.mean(err_mtx_rnn[:, 13:])
    mean_err_birnn_velo = np.mean(err_mtx_birnn[:, 13:])
    mean_err_dkf_velo = np.mean(err_mtx_dkf_velo)
    std_err_rnn_disp = np.std(err_mtx_rnn[:, 0:13])
    std_err_birnn_disp = np.std(err_mtx_birnn[:, 0:13])
    std_err_dkf_disp = np.std(err_mtx_dkf_disp)
    std_err_rnn_velo = np.std(err_mtx_rnn[:, 13:])
    std_err_birnn_velo = np.std(err_mtx_birnn[:, 13:])
    std_err_dkf_velo = np.std(err_mtx_dkf_velo)
    plt.figure(figsize=(10 * cm, 8 * cm))
    plt.bar(
        np.arange(3),
        [mean_err_rnn_disp*100, mean_err_birnn_disp*100, mean_err_dkf_disp*100],
        yerr=[std_err_rnn_disp*100, std_err_birnn_disp*100, std_err_dkf_disp*100],
        capsize=5,
        color="b",
    )
    plt.ylim(0, 100)
    plt.xticks([0, 1, 2], ["RNN", "BiRNN", "DKF"])
    plt.ylabel("Similarity (%)")
    plt.title("Displacement")
    plt.savefig("./figures/performance_disp.svg", bbox_inches="tight")
    plt.show()
    plt.figure(figsize=(10 * cm, 8 * cm))
    plt.bar(
        np.arange(3),
        [mean_err_rnn_velo*100, mean_err_birnn_velo*100, mean_err_dkf_velo*100],
        yerr=[std_err_rnn_velo*100, std_err_birnn_velo*100, std_err_dkf_velo*100],
        capsize=5,
        color="r",
    )
    plt.xticks([0, 1, 2], ["RNN", "BiRNN", "DKF"])
    plt.ylabel("Similarity (%)")
    plt.title("Velocity")
    plt.savefig("./figures/performance_velo.svg", bbox_inches="tight")
    plt.show()


