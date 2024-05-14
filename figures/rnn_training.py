import numpy as np
import matplotlib.pyplot as plt
import torch
from models import Rnn
from exps import shear_type_structure
import pickle
from utils import similarity

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 8})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def loss_curve():
    loss_save_path = "./dataset/sts/birnn.pkl"
    with open(loss_save_path, "rb") as f:
        birnn = torch.load(f)
    loss_save_path = "./dataset/sts/rnn.pkl"
    with open(loss_save_path, "rb") as f:
        rnn = torch.load(f)
    epoch_birnn = np.arange(1, (len(birnn["train_loss_list"])) * 200, 200)
    epoch_rnn = np.arange(1, (len(rnn["train_loss_list"])) * 200, 200)
    plt.figure(figsize=(9 * cm, 7 * cm))
    plt.plot(epoch_rnn, rnn["train_loss_list"], color="b", linewidth=1.2)
    plt.plot(epoch_rnn, rnn["test_loss_list"], color="b", linestyle="--", linewidth=1.2)
    plt.plot(epoch_birnn, birnn["train_loss_list"], color="r", linewidth=1.2)
    plt.plot(
        epoch_birnn, birnn["test_loss_list"], color="r", linestyle="--", linewidth=1.2
    )

    plt.tick_params(which="both", direction="in")
    plt.legend(
        ["RNN training", "RNN test", "BiRNN training", "BiRNN test"],
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
    )
    plt.xlim(0, 80000)
    plt.xticks(
        [0, 20000, 40000, 60000, 80000],
        ["0", "2", "4", "6", "8"],
        fontsize=8,
    )
    plt.yscale("log")
    plt.xlabel(r"Epoch ($\times$10$^4$)", fontsize=8)
    plt.ylabel("Loss", fontsize=8)
    plt.grid(True)
    # plt.savefig("./figures/loss_curve.svg", bbox_inches="tight")
    plt.savefig("./figures/F_loss_curve.pdf", bbox_inches="tight")
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
    time = np.arange(0, 40, 1 / 20)
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    dkf_pred = dkf_pred["disp_pred"]
    with open("./dataset/sts/akf_pred.pkl", "rb") as f:
        akf_pred = pickle.load(f)
    akf_pred = akf_pred["disp_pred"]
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
    akf_state_pred = akf_pred[:, :, dof_idx]
    akf_state_pred[file_idx, 1:] = akf_state_pred[file_idx, :-1]
    akf_state_pred[file_idx, 0] = 0
    plt.figure(figsize=(20 * cm, 8 * cm))
    plt.plot(
        time, state_test[file_idx, :] * 100, label="Ref.", color="k", linewidth=1.2
    )
    plt.plot(
        time,
        birnn_state_pred[file_idx, :] * 100,
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    plt.plot(
        time,
        rnn_state_pred[file_idx, :] * 100,
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    plt.plot(
        time,
        dkf_state_pred[file_idx, :] * 100,
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    plt.plot(
        time,
        akf_state_pred[file_idx, :] * 100,
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )
    plt.legend(fontsize=8, facecolor="white", edgecolor="black", ncol=3)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (cm)")
    plt.tick_params(which="both", direction="in")
    plt.xlim(0, 40)
    plt.ylim(-0.5 * 100, 0.5 * 100)
    plt.text(
        0.9, 0.125, "9th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.savefig("./figures/disp_pred.svg", bbox_inches="tight")
    plt.savefig("./figures/F_disp_pred.pdf", bbox_inches="tight")
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
    time = np.arange(0, 40, 1 / 20)
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    with open("./dataset/sts/akf_pred.pkl", "rb") as f:
        akf_pred = pickle.load(f)

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
    akf_pred = akf_pred["velo_pred"]
    akf_state_pred = akf_pred[:, :, dof_idx]
    akf_state_pred[file_idx, 1:] = akf_state_pred[file_idx, :-1]
    akf_state_pred[file_idx, 0] = 0

    plt.figure(figsize=(20 * cm, 8 * cm))
    plt.plot(time, state_test[file_idx, :], label="Ref.", color="k", linewidth=1.2)
    plt.plot(
        time,
        birnn_state_pred[file_idx, :],
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    plt.plot(
        time,
        rnn_state_pred[file_idx, :],
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    plt.plot(
        time,
        dkf_state_pred[file_idx, :],
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    plt.plot(
        time,
        akf_state_pred[file_idx, :],
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )
    plt.legend(fontsize=8, facecolor="white", edgecolor="black", ncol=3)
    plt.text(
        0.9,
        0.125,
        "13th floor",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.tick_params(which="both", direction="in")
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.savefig("./figures/velo_pred.svg", bbox_inches="tight")
    plt.savefig("./figures/F_velo_pred.pdf", bbox_inches="tight")
    plt.show()


def state_pred():
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
    time = np.arange(0, 40, 1 / 20)
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    dkf_pred = dkf_pred["disp_pred"]
    with open("./dataset/sts/akf_pred.pkl", "rb") as f:
        akf_pred = pickle.load(f)
    akf_pred = akf_pred["disp_pred"]
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
    akf_state_pred = akf_pred[:, :, dof_idx]
    akf_state_pred[file_idx, 1:] = akf_state_pred[file_idx, :-1]
    akf_state_pred[file_idx, 0] = 0

    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 12 * cm))
    axs[0].plot(
        time, state_test[file_idx, :] * 100, label="Ref.", color="k", linewidth=1.2
    )
    axs[0].plot(
        time,
        rnn_state_pred[file_idx, :] * 100,
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    axs[0].plot(
        time,
        birnn_state_pred[file_idx, :] * 100,
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    axs[0].plot(
        time,
        dkf_state_pred[file_idx, :] * 100,
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    axs[0].plot(
        time,
        akf_state_pred[file_idx, :] * 100,
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )
    # axs[0].legend(fontsize=8, facecolor="white", edgecolor="black", ncol=3)
    axs[0].text(
        0.8, 0.125, "9th floor", ha="center", va="center", transform=axs[0].transAxes
    )
    axs[0].text(
        -0.1 / 3, -0.1, "(a)", ha="center", va="center", transform=axs[0].transAxes
    )
    axs[0].grid(True)
    # axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Displacement (cm)")
    axs[0].tick_params(which="both", direction="in")
    axs[0].set_xlim(0, 40)
    axs[0].set_ylim(-0.4 * 100, 0.4 * 100)
    axs[0].set_yticks([-40, -20, 0, 20, 40])
    axs[0].set_xticks([0, 10, 20, 30, 40])
    fig.legend(
        bbox_to_anchor=(0.5, 0.95),
        loc="outside upper center",
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        ncol=5,
    )
    x1, x2, y1, y2 = 11.5, 12.7, -20, -4.5
    axins = axs[0].inset_axes(
        [0.01, 0.03, 0.26, 0.26],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[],
    )
    axins.plot(
        time,
        state_test[file_idx, :] * 100,
        label="Ref.",
        color="k",
        linewidth=1.2,
    )
    axins.plot(
        time,
        rnn_state_pred[file_idx, :] * 100,
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    axins.plot(
        time,
        birnn_state_pred[file_idx, :] * 100,
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    axins.plot(
        time,
        dkf_state_pred[file_idx, :] * 100,
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    axins.plot(
        time,
        akf_state_pred[file_idx, :] * 100,
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )
    axins.set_xticks([])
    axins.set_yticks([])
    axs[0].indicate_inset_zoom(axins, edgecolor="black")

    with open("./dataset/sts/dkf_pred.pkl", "rb") as f:
        dkf_pred = pickle.load(f)
    with open("./dataset/sts/akf_pred.pkl", "rb") as f:
        akf_pred = pickle.load(f)

    _, _, state_test, acc_test = shear_type_structure.training_test_data(
        [0, 1, 2, 3, 4], 1, 90, 100
    )
    with torch.no_grad():
        rnn_state_pred, _ = RNN4ststate(acc_test, rnn_test_h0)
        birnn_state_pred, _ = BiRNN4ststate(acc_test, birnn_test_h0)
    dof_idx = -1

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
    akf_pred = akf_pred["velo_pred"]
    akf_state_pred = akf_pred[:, :, dof_idx]
    akf_state_pred[file_idx, 1:] = akf_state_pred[file_idx, :-1]
    akf_state_pred[file_idx, 0] = 0

    axs[1].plot(time, state_test[file_idx, :], label="Ref.", color="k", linewidth=1.2)
    axs[1].plot(
        time,
        rnn_state_pred[file_idx, :],
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    axs[1].plot(
        time,
        birnn_state_pred[file_idx, :],
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    axs[1].plot(
        time,
        dkf_state_pred[file_idx, :],
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    axs[1].plot(
        time,
        akf_state_pred[file_idx, :],
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )

    axs[1].text(
        0.8,
        0.125,
        "13th floor",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].text(
        -0.1 / 3,
        -0.1,
        "(b)",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].grid(True)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].tick_params(which="both", direction="in")
    axs[1].set_xlim(0, 40)
    axs[1].set_ylim(-2, 2)
    axs[1].set_xticks([0, 10, 20, 30, 40])
    axs[1].set_yticks([-2, -1, 0, 1, 2])
    x1, x2, y1, y2 = 12.9, 14.2, -0.98, -0.16
    axins2 = axs[1].inset_axes(
        [0.01, 0.03, 0.26, 0.26],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[],
    )
    axins2.plot(
        time,
        state_test[file_idx, :],
        label="Ref.",
        color="k",
        linewidth=1.2,
    )
    axins2.plot(
        time,
        rnn_state_pred[file_idx, :],
        label="RNN pred.",
        linestyle="-.",
        color="b",
        linewidth=1.2,
    )
    axins2.plot(
        time,
        birnn_state_pred[file_idx, :],
        label="BiRNN pred.",
        linestyle="--",
        color="r",
        linewidth=1.2,
    )
    axins2.plot(
        time,
        dkf_state_pred[file_idx, :],
        label="DKF pred.",
        linestyle=":",
        color="lime",
        linewidth=1.2,
    )
    axins2.plot(
        time,
        akf_state_pred[file_idx, :],
        label="AKF pred.",
        linestyle="--",
        color="darkviolet",
        linewidth=1.2,
    )
    axins2.set_xticks([])
    axins2.set_yticks([])
    axs[1].indicate_inset_zoom(axins2, edgecolor="black")

    plt.savefig("./figures/state_pred.svg", bbox_inches="tight")
    plt.savefig("./figures/F_state_pred.pdf", bbox_inches="tight")
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
    with open("./dataset/sts/akf_pred.pkl", "rb") as f:
        akf_pred = pickle.load(f)
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
    dkf_pred_disp[:, 1:, :] = dkf_pred_disp[:, :-1, :]
    dkf_pred_disp[:, 0, :] = 0
    dkf_pred_velo[:, 1:, :] = dkf_pred_velo[:, :-1, :]
    dkf_pred_velo[:, 0, :] = 0
    akf_pred_disp = akf_pred["disp_pred"]
    akf_pred_velo = akf_pred["velo_pred"]
    akf_pred_disp[:, 1:, :] = akf_pred_disp[:, :-1, :]
    akf_pred_disp[:, 0, :] = 0
    akf_pred_velo[:, 1:, :] = akf_pred_velo[:, :-1, :]
    akf_pred_velo[:, 0, :] = 0
    err_mtx_rnn = similarity(rnn_state_pred, state_test)
    err_mtx_birnn = similarity(birnn_state_pred, state_test)
    err_mtx_dkf_disp = similarity(dkf_pred_disp, state_test)
    err_mtx_dkf_velo = similarity(dkf_pred_velo, state_test[:, :, 13:])
    err_mtx_akf_disp = similarity(akf_pred_disp, state_test)
    err_mtx_akf_velo = similarity(akf_pred_velo, state_test[:, :, 13:])
    mean_err_rnn_disp = np.mean(err_mtx_rnn[:, 0:13])
    mean_err_birnn_disp = np.mean(err_mtx_birnn[:, 0:13])
    mean_err_dkf_disp = np.mean(err_mtx_dkf_disp)
    mean_err_rnn_velo = np.mean(err_mtx_rnn[:, 13:])
    mean_err_birnn_velo = np.mean(err_mtx_birnn[:, 13:])
    mean_err_dkf_velo = np.mean(err_mtx_dkf_velo)
    mean_err_akf_disp = np.mean(err_mtx_akf_disp)
    mean_err_akf_velo = np.mean(err_mtx_akf_velo)
    # std_err_rnn_disp = np.std(err_mtx_rnn[:, 0:13])
    # std_err_birnn_disp = np.std(err_mtx_birnn[:, 0:13])
    # std_err_dkf_disp = np.std(err_mtx_dkf_disp)
    # std_err_rnn_velo = np.std(err_mtx_rnn[:, 13:])
    # std_err_birnn_velo = np.std(err_mtx_birnn[:, 13:])
    # std_err_dkf_velo = np.std(err_mtx_dkf_velo)
    # std_err_akf_disp = np.std(err_mtx_akf_disp)
    # std_err_akf_velo = np.std(err_mtx_akf_velo)
    fig, axs = plt.subplots(1, 2, figsize=(18 * cm, 8 * cm))

    # axs[0].set_ylim(0, 100)
    axs[0].set_xticks([0, 1, 2, 3], ["RNN", "BiRNN", "DKF", "AKF"])
    axs[0].set_ylabel("NRMSE")
    axs[0].title.set_text("Displacement")
    axs[0].tick_params(which="both", direction="in")
    axs[0].text(
        -0.1, -0.06, "(a)", ha="center", va="center", transform=axs[0].transAxes
    )
    axs[0].set_ylim(0, 0.2)
    axs[0].set_yticks([0, 0.1, 0.2])
    axs[0].grid(True)

    axs[0].bar(
        np.arange(4),
        [
            mean_err_rnn_disp,
            mean_err_birnn_disp,
            mean_err_dkf_disp,
            mean_err_akf_disp,
        ],
        # yerr=[
        #     std_err_rnn_disp,
        #     std_err_birnn_disp,
        #     std_err_dkf_disp,
        #     std_err_akf_disp,
        # ],
        # capsize=5,
        color="b",
        zorder=3,

    )
    # axs[0].savefig("./figures/performance_disp.svg", bbox_inches="tight")
    # plt.show()
    # plt.figure(figsize=(10 * cm, 8 * cm))

    axs[1].set_xticks([0, 1, 2, 3], ["RNN", "BiRNN", "DKF", "AKF"])
    axs[1].set_ylabel("NRMSE")
    axs[1].title.set_text("Velocity")
    axs[1].tick_params(which="both", direction="in")
    axs[1].text(
        -0.1,
        -0.06,
        "(b)",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].set_ylim(0, 0.3)
    axs[1].set_yticks([0, 0.1, 0.2, 0.3])
    axs[1].grid(True)
    axs[1].bar(
        np.arange(4),
        [
            mean_err_rnn_velo,
            mean_err_birnn_velo,
            mean_err_dkf_velo,
            mean_err_akf_velo,
        ],
        # yerr=[
        #     std_err_rnn_velo,
        #     std_err_birnn_velo,
        #     std_err_dkf_velo,
        #     std_err_akf_velo,
        # ],
        # capsize=5,
        color="r",
        zorder=3,
    )
    plt.savefig("./figures/performance.svg", bbox_inches="tight")
    plt.savefig("./figures/F_performance.pdf", bbox_inches="tight")
    plt.show()
