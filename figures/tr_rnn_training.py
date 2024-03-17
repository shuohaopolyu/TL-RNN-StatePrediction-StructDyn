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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def disp_birnn_pred(which=0):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    acc_list, state_list = shear_type_structure.generate_seismic_response(
        acc_sensor, num_seismic
    )
    output_size = 26
    acc_tensor = acc_list[which].unsqueeze(0)
    state_tensor = state_list[which].squeeze(0)
    state_tensor = state_tensor.cpu().numpy()
    print(state_tensor.shape)
    h0 = torch.zeros(2, 1, 30).to(device)
    h1 = torch.zeros(1, 1, 30).to(device)
    trbirnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    birnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/sts/tr_birnn00" + str(which) + ".pth", "rb") as f:
        trbirnn.load_state_dict(torch.load(f))
    trbirnn.eval()
    with torch.no_grad():
        output, _ = trbirnn(acc_tensor, h0)
    output = output.squeeze(0).cpu().numpy()

    with open("./dataset/sts/birnn.pth", "rb") as f:
        birnn.load_state_dict(torch.load(f))
    birnn.eval()
    with torch.no_grad():
        output2, _ = birnn(acc_tensor, h0)
    output2 = output2.squeeze(0).cpu().numpy()

    time = np.arange(0, output.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(1, 1, figsize=(20 * cm, 8 * cm))
    ax.plot(time, state_tensor[:, 7] * 100, label="Ref.", color="k", linewidth=1.2)
    ax.plot(
        time,
        output[:, 7] * 100,
        label="TL-BiRNN pred.",
        color="r",
        linewidth=1.2,
        linestyle="-.",
    )
    ax.plot(
        time,
        output2[:, 7] * 100,
        label="BiRNN pred.",
        color="b",
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_ylim(-20, 20)

    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=3)
    ax.set_xlim(0, 40)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-20, 20, 10))
    ax.text(
        0.1, 0.875, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    plt.savefig("./figures/tr_birnn_disp.svg", bbox_inches="tight")
    plt.show()


def disp_rnn_pred(which=0):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    acc_list, state_list = shear_type_structure.generate_seismic_response(
        acc_sensor, num_seismic
    )
    output_size = 26
    acc_tensor = acc_list[which].unsqueeze(0)
    state_tensor = state_list[which].squeeze(0)
    state_tensor = state_tensor.cpu().numpy()
    h1 = torch.zeros(1, 1, 30).to(device)
    trrnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=False,
    )
    rnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=False,
    )

    with open("./dataset/sts/tr_rnn00" + str(which) + ".pth", "rb") as f:
        trrnn.load_state_dict(torch.load(f))
    trrnn.eval()
    with torch.no_grad():
        output3, _ = trrnn(acc_tensor, h1)
    output3 = output3.squeeze(0).cpu().numpy()
    with open("./dataset/sts/rnn.pth", "rb") as f:
        rnn.load_state_dict(torch.load(f))
    rnn.eval()
    with torch.no_grad():
        output4, _ = rnn(acc_tensor, h1)
    output4 = output4.squeeze(0).cpu().numpy()

    # output5, _ = shear_type_structure.dkf_seismic_pred()
    # output5 = output5[which]
    time = np.arange(0, output3.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(1, 1, figsize=(20 * cm, 8 * cm))
    ax.plot(time, state_tensor[:, 7] * 100, label="Ref.", color="k", linewidth=1.2)
    ax.plot(
        time,
        output3[:, 7] * 100,
        label="TL-RNN pred.",
        color="r",
        linewidth=1.2,
        linestyle="-.",
    )
    ax.plot(
        time,
        output4[:, 7] * 100,
        label="RNN pred.",
        color="b",
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_ylim(-20, 60)
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=4)
    ax.set_xlim(0, 40)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-20, 61, 20))
    ax.text(
        0.1, 0.875, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    plt.savefig("./figures/tr_rnn_disp.svg", bbox_inches="tight")
    plt.show()


def disp_kf_pred(which=0):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    _, state_list = shear_type_structure.generate_seismic_response(
        acc_sensor, num_seismic
    )

    state_tensor = state_list[which].squeeze(0)
    state_tensor = state_tensor.cpu().numpy()

    disp1, _ = shear_type_structure.dkf_seismic_pred()
    disp1 = disp1[which]
    disp1[1:, :] = disp1[:-1, :]
    disp1[:, 0] = 0
    disp2, _ = shear_type_structure.integr_dkf_seismic_pred()
    disp2 = disp2[which]
    disp2[1:, :] = disp2[:-1, :]
    disp2[:, 0] = 0
    disp3, _ = shear_type_structure.akf_seismic_pred()
    disp3 = disp3[which]
    disp3[1:, :] = disp3[:-1, :]
    disp3[:, 0] = 0
    disp4, _ = shear_type_structure.integr_akf_seismic_pred()
    disp4 = disp4[which]
    disp4[1:, :] = disp4[:-1, :]
    disp4[:, 0] = 0
    time = np.arange(0, disp1.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(1, 1, figsize=(20 * cm, 8 * cm))
    plt.plot(time, state_tensor[:, 7] * 100, label="Ref.", color="k", linewidth=1.2)
    ax.plot(
        time,
        disp1[:, 7] * 100,
        label="DKF pred.",
        color="r",
        linewidth=1.2,
        linestyle="-.",
    )
    ax.plot(
        time,
        disp2[:, 7] * 100,
        label="Integr. DKF pred.",
        color="b",
        linewidth=1.2,
        linestyle="--",
    )
    ax.plot(
        time,
        disp3[:, 7] * 100,
        label="AKF pred.",
        color="lime",
        linewidth=1.2,
        linestyle=":",
    )
    ax.plot(
        time,
        disp4[:, 7] * 100,
        label="Integr. AKF pred.",
        color="m",
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_ylim(-20, 40)
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=3)
    ax.set_xlim(0, 40)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-20, 41, 20))
    ax.text(
        0.1, 0.125, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    plt.savefig("./figures/tr_kf_disp.svg", bbox_inches="tight")
    plt.show()


def velo_pred(which=1):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    acc_list, state_list = shear_type_structure.generate_seismic_response(
        acc_sensor, num_seismic
    )
    output_size = 26
    dof = 8
    acc_tensor = acc_list[which].unsqueeze(0)
    state_tensor = state_list[which].squeeze(0)
    state_tensor = state_tensor.cpu().numpy()
    h0 = torch.zeros(2, 1, 30).to(device)
    h1 = torch.zeros(1, 1, 30).to(device)
    trbirnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    birnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    trrnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=False,
    )
    rnn = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=False,
    )
    with open("./dataset/sts/tr_birnn00" + str(which) + ".pth", "rb") as f:
        trbirnn.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        birnn.load_state_dict(torch.load(f))
    with open("./dataset/sts/tr_rnn00" + str(which) + ".pth", "rb") as f:
        trrnn.load_state_dict(torch.load(f))
    with open("./dataset/sts/rnn.pth", "rb") as f:
        rnn.load_state_dict(torch.load(f))
    with torch.no_grad():
        output, _ = trbirnn(acc_tensor, h0)
        output2, _ = birnn(acc_tensor, h0)
        output3, _ = trrnn(acc_tensor, h1)
        output4, _ = rnn(acc_tensor, h1)
    output = output.squeeze(0).cpu().numpy()
    output2 = output2.squeeze(0).cpu().numpy()
    output3 = output3.squeeze(0).cpu().numpy()
    output4 = output4.squeeze(0).cpu().numpy()
    _, output5 = shear_type_structure.dkf_seismic_pred()

    output5 = output5[which]
    output5[1:, :] = output5[:-1, :]
    output5[:, 0] = 0
    _, output6 = shear_type_structure.akf_seismic_pred()
    output6 = output6[which]
    output6[1:, :] = output6[:-1, :]
    output6[:, 0] = 0
    time = np.arange(0, output.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(1, 1, figsize=(20 * cm, 8 * cm))
    ax.plot(
        time, state_tensor[:, dof + 13] * 100, label="Ref.", color="k", linewidth=1.2
    )
    ax.plot(
        time,
        output2[:, dof + 13] * 100,
        label="BiRNN pred.",
        color="r",
        linewidth=0.8,
        linestyle="--",
    )
    ax.plot(
        time,
        output4[:, dof + 13] * 100,
        label="RNN pred.",
        color="b",
        linewidth=0.8,
        linestyle=":",
    )
    ax.plot(
        time,
        output5[:, dof] * 100,
        label="DKF pred.",
        color="orange",
        linewidth=0.8,
        linestyle="-.",
    )
    ax.plot(
        time,
        output6[:, dof] * 100,
        label="AKF pred.",
        color="m",
        linewidth=0.8,
        linestyle="-.",
    )
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=3)
    ax.set_xlim(0, 70)
    ax.set_ylim(-20, 20)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-20, 21, 10))
    ax.text(
        0.1,
        0.875,
        "11th floor",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (cm/s)")
    plt.savefig("./figures/tr_velo.svg", bbox_inches="tight")
    plt.show()


def loss_curve():
    # training loss
    epoch_list = [[4000, 5000], [7000, 10000], [3800, 6000], [4000, 10000]]
    # epoch_list = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]

    for which in range(4):
        comp_rate = 1
        path = "./dataset/sts/tr_birnn" + format(which, "03") + ".pkl"
        with open(path, "rb") as f:
            tr_birnn_loss = pickle.load(f)
        tr_birnn_loss = tr_birnn_loss[: epoch_list[which][1]]
        tr_birnn_loss = np.array(tr_birnn_loss)
        birnn_train_loss = tr_birnn_loss[::comp_rate, 0]
        birnn_test_loss = tr_birnn_loss[::comp_rate, 1]
        path = "./dataset/sts/tr_rnn" + format(which, "03") + ".pkl"
        with open(path, "rb") as f:
            tr_rnn_loss = pickle.load(f)
        tr_rnn_loss = tr_rnn_loss[: epoch_list[which][0]]
        tr_rnn_loss = np.array(tr_rnn_loss)
        rnn_train_loss = tr_rnn_loss[::comp_rate, 0]
        rnn_test_loss = tr_rnn_loss[::comp_rate, 1]
        time_birnn = np.arange(0, len(birnn_train_loss) * comp_rate, comp_rate)
        time = np.arange(0, len(rnn_train_loss) * comp_rate, comp_rate)
        fig, ax = plt.subplots(1, 1, figsize=(10 * cm, 8 * cm))
        ax.plot(
            time_birnn,
            birnn_train_loss,
            label="BiRNN training",
            color="b",
            linewidth=1.5,
        )
        ax.plot(
            time_birnn,
            birnn_test_loss,
            label="BiRNN test",
            color="b",
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(time, rnn_train_loss, label="RNN training", color="r", linewidth=1.5)
        ax.plot(
            time,
            rnn_test_loss,
            label="RNN test",
            color="r",
            linestyle="--",
            linewidth=1.5,
        )
        ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=1)
        # ax.set_xlim(-1, 5000)
        # ax.set_ylim(0, 0.1)
        ax.grid(True)
        ax.tick_params(which="both", direction="in")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        plt.savefig(
            "./figures/tr_birnn_loss" + str(which) + ".svg", bbox_inches="tight"
        )
        plt.show()


def performance_evaluation():
    pass
