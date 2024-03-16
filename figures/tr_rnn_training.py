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


def disp_birnn_pred(which=3):
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
    ax.set_ylim(-15, 15)

    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=3)
    ax.set_xlim(0, 50)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-15, 16, 5))
    ax.text(
        0.1, 0.875, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    if type == "tr_birnn":
        plt.savefig("./figures/tr_birnn_disp.svg", bbox_inches="tight")
    plt.show()


def disp_rnn_dkf_pred(which=3):
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
    # ax.plot(
    #     time,
    #     output5[:, 7] * 100,
    #     label="DKF pred.",
    #     color="g",
    #     linewidth=1.2,
    #     linestyle=":",
    # )
    ax.set_ylim(-15, 60)
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=4)
    ax.set_xlim(0, 50)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-15, 61, 15))
    ax.text(
        0.1, 0.875, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    plt.savefig("./figures/tr_rnn_disp.svg", bbox_inches="tight")
    plt.show()


def disp_kf_pred(which=3):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    acc_list, state_list = shear_type_structure.generate_seismic_response(
        acc_sensor, num_seismic
    )

    state_tensor = state_list[which].squeeze(0)
    state_tensor = state_tensor.cpu().numpy()

    disp1, _ = shear_type_structure.dkf_seismic_pred()
    disp1 = disp1[which]
    disp2, _ = shear_type_structure.exp_dkf_seismic_pred()
    disp2 = disp2[which]
    disp3, _ = shear_type_structure.akf_seismic_pred()
    disp3 = disp3[which]
    disp4, _ = shear_type_structure.exp_akf_seismic_pred()
    disp4 = disp4[which]
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
        label="Exp. DKF pred.",
        color="b",
        linewidth=1.2,
        linestyle="--",
    )
    ax.plot(
        time,
        disp3[:, 7] * 100,
        label="AKF pred.",
        color="g",
        linewidth=1.2,
        linestyle=":",
    )
    ax.plot(
        time,
        disp4[:, 7] * 100,
        label="Exp. AKF pred.",
        color="m",
        linewidth=1.2,
        linestyle="--",
    )
    # ax.set_ylim(-15, 60)
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=5)
    ax.set_xlim(0, 50)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    # ax.set_yticks(np.arange(-15, 61, 15))
    ax.text(
        0.1, 0.875, "8th floor", ha="center", va="center", transform=plt.gca().transAxes
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (cm)")
    plt.savefig("./figures/tr_kf_disp.svg", bbox_inches="tight")
    plt.show()


def velo_pred(which=3):
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
    time = np.arange(0, output.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(1, 1, figsize=(20 * cm, 8 * cm))
    ax.plot(
        time, state_tensor[:, dof + 13] * 100, label="Ref.", color="k", linewidth=1.2
    )
    # ax.plot(
    #     time,
    #     output[:, dof + 13] * 100,
    #     label="TL-BiRNN pred.",
    #     color="r",
    #     linewidth=1.2,
    #     linestyle="-.",
    # )
    ax.plot(
        time,
        output2[:, dof + 13] * 100,
        label="BiRNN pred.",
        color="r",
        linewidth=1.2,
        linestyle="--",
    )
    # ax.plot(
    #     time,
    #     output3[:, dof + 13] * 100,
    #     label="TL-RNN pred.",
    #     color="g",
    #     linewidth=0.8,
    #     linestyle=":",
    # )
    ax.plot(
        time,
        output4[:, dof + 13] * 100,
        label="RNN pred.",
        color="b",
        linewidth=1.2,
        linestyle="-.",
    )
    ax.plot(
        time,
        output5[:, dof] * 100,
        label="DKF pred.",
        color="g",
        linewidth=1.2,
        linestyle=":",
    )
    ax.legend(fontsize=10, facecolor="white", edgecolor="black", ncol=4)
    ax.set_xlim(0, 50)
    ax.grid(True)
    ax.tick_params(which="both", direction="in")
    ax.set_yticks(np.arange(-15, 61, 15))
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
    pass
