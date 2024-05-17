import numpy as np
import matplotlib.pyplot as plt
import pickle
from exps import continuous_beam as cb
import scipy.io
import scipy.signal
import torch


# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 8})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def ema():
    data_path = "./dataset/csb/noise_for_ema_2.mat"
    exp_data = scipy.io.loadmat(data_path)
    acc1 = -exp_data["Data1_AI_1_AI_1"][2000:7000, 0].reshape(-1, 1)
    acc2 = -exp_data["Data1_AI_2_AI_2"][2000:7000, 0].reshape(-1, 1)
    acc3 = -exp_data["Data1_AI_3_AI_3"][2000:7000, 0].reshape(-1, 1)
    force = exp_data["Data1_AI_4_AI_4"][2000:7000, 0].reshape(-1, 1)
    time = np.arange(0, 0.0002 * len(acc1), 0.0002)
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 8 * cm))
    axs[0, 0].set_ylabel("Acceleration (g)")
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].set_ylim(-3, 3)
    axs[0, 0].set_yticks([-3, -1.5, 0, 1.5, 3])
    axs[0, 0].tick_params(axis="both", direction="in", which="both")
    axs[0, 0].grid(True)
    axs[0, 0].plot(time, acc3, color="black", linewidth=0.6)
    axs[0, 0].text(
        -0.1,
        -0.1,
        "(a)",
        ha="center",
        va="center",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 0].text(
        0.9,
        0.9,
        "A1",
        ha="center",
        va="center",
        transform=axs[0, 0].transAxes,
    )

    axs[0, 1].set_ylabel("Acceleration (g)")
    axs[0, 1].set_xlim(0, 1)
    axs[0, 1].set_ylim(-3, 3)
    axs[0, 1].set_yticks([-3, -1.5, 0, 1.5, 3])
    axs[0, 1].tick_params(axis="both", direction="in", which="both")
    axs[0, 1].grid(True)
    axs[0, 1].plot(time, acc2, color="black", linewidth=0.6)
    axs[0, 1].text(
        -0.1,
        -0.1,
        "(b)",
        ha="center",
        va="center",
        transform=axs[0, 1].transAxes,
    )
    axs[0, 1].text(
        0.9,
        0.9,
        "A2",
        ha="center",
        va="center",
        transform=axs[0, 1].transAxes,
    )

    axs[1, 0].set_ylabel("Acceleration (g)")
    axs[1, 0].set_xlim(0, 1)

    axs[1, 0].set_ylim(-4, 4)
    axs[1, 0].set_yticks([-4, -2, 0, 2, 4], ["-4.0", "-2.0", "0.0", "2.0", "4.0"])
    axs[1, 0].tick_params(axis="both", direction="in", which="both")
    axs[1, 0].grid(True)
    axs[1, 0].plot(time, acc1, color="black", linewidth=0.6)
    axs[1, 0].text(
        -0.1,
        -0.1,
        "(c)",
        ha="center",
        va="center",
        transform=axs[1, 0].transAxes,
    )
    axs[1, 0].text(
        0.9,
        0.9,
        "A3",
        ha="center",
        va="center",
        transform=axs[1, 0].transAxes,
    )

    axs[1, 1].set_ylabel("Force (N)")
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_ylim(-2, 2)
    axs[1, 1].set_yticks([-2, -1, 0, 1, 2], ["-2.0", "-1.0", "0.0", "1.0", "2.0"])
    axs[1, 1].tick_params(axis="both", direction="in", which="both")
    axs[1, 1].grid(True)
    axs[1, 1].plot(time, force, color="black", linewidth=0.6)
    axs[1, 1].text(
        -0.1,
        -0.1,
        "(d)",
        ha="center",
        va="center",
        transform=axs[1, 1].transAxes,
    )

    fig.supxlabel("Time (s)", fontsize=8)
    plt.savefig("./figures/F_csb_ema.pdf", bbox_inches="tight")
    plt.show()


def model_updating():
    result_path = "./dataset/csb/model_updating.pkl"
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    frf_data_path = "./dataset/csb/ema.pkl"
    with open(frf_data_path, "rb") as f:
        frf_data = pickle.load(f)
    # print(result.x)
    # cb.compare_frf(result, frf_data)
    frf_model = cb.compute_frf(*(result.x))
    phase_data = np.unwrap(np.abs(np.angle(frf_data)))
    phase_mtx = np.unwrap(np.abs(np.angle(frf_model)))
    amp_data = np.abs(frf_data) / 9.8
    amp_mtx = np.abs(frf_model) / 9.8
    # interchange the first and the third column of the data
    amp_data[:, [0, 2]] = amp_data[:, [2, 0]]
    phase_data[:, [0, 2]] = phase_data[:, [2, 0]]
    amp_mtx[:, [0, 2]] = amp_mtx[:, [2, 0]]
    phase_mtx[:, [0, 2]] = phase_mtx[:, [2, 0]]

    freq = np.linspace(10, 100, 450)
    figidx = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fig, axs = plt.subplots(2, 3, figsize=(18 * cm, 10 * cm))
    axs[0, 0].plot(freq, amp_data[:, 0], color="red", linewidth=1.2)
    axs[0, 0].plot(freq, amp_mtx[:, 0], color="blue", linestyle="--", linewidth=1.2)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel(
        r"Magnitude (g/N)",
    )

    axs[1, 0].plot(freq, phase_data[:, 0], color="red", linewidth=1.2)
    axs[1, 0].plot(freq, phase_mtx[:, 0], color="blue", linestyle="--", linewidth=1.2)
    axs[1, 0].set_ylabel("Phase (rad)", labelpad=10)

    axs[0, 1].plot(
        freq, amp_data[:, 1], color="red", linewidth=1.2, label="Measurements"
    )
    axs[0, 1].plot(
        freq, amp_mtx[:, 1], color="blue", linestyle="--", linewidth=1.2, label="Model"
    )
    axs[0, 1].set_yscale("log")
    # axs[0, 1].set_ylabel(r"Magnitude (m/s$^2$/N)")
    axs[1, 1].plot(freq, phase_data[:, 1], color="red", linewidth=1.2)
    axs[1, 1].plot(freq, phase_mtx[:, 1], color="blue", linestyle="--", linewidth=1.2)
    # axs[1, 1].set_ylabel("Phase (rad)")
    axs[0, 2].plot(freq, amp_data[:, 2], color="red", linewidth=1.2)
    axs[0, 2].plot(freq, amp_mtx[:, 2], color="blue", linestyle="--", linewidth=1.2)
    axs[0, 2].set_yscale("log")
    # axs[0, 2].set_ylabel(r"Magnitude (m/s$^2$/N)")
    axs[1, 2].plot(freq, phase_data[:, 2], color="red", linewidth=1.2)
    axs[1, 2].plot(freq, phase_mtx[:, 2], color="blue", linestyle="--", linewidth=1.2)
    # axs[1, 2].set_ylabel("Phase (rad)")
    axs[1, 1].set_xlabel("Frequency (Hz)")

    for j in range(3):
        for i in range(2):
            axs[i, j].tick_params(axis="y", direction="in", which="both")
            axs[i, j].set_xlim(10, 100)
            axs[1, j].set_ylim(0, 3.15)
            # axs[i, 0].set_ylim(1, 1000)
            axs[i, j].grid(True)
            axs[i, j].text(
                -0.1,
                -0.1,
                figidx[i * 3 + j],
                ha="center",
                va="center",
                transform=axs[i, j].transAxes,
            )
            axs[i, j].tick_params(axis="x", direction="in", which="both")
            axs[i, j].set_xticks([10, 20, 40, 60, 80, 100])
            axs[i, j].set_xticklabels(["10", "20", "40", "60", "80", "100"])
        # let legend outside the plot
    axs[0, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        ncol=2,
    )
    plt.tight_layout()

    plt.savefig("./figures/F_csb_model_updating.pdf", bbox_inches="tight")
    plt.show()


def filter_disp():
    # This figure aims to show that the laser displacement
    # sensor data is accurate within a certain frequency range
    filename = f"./dataset/csb/exp_" + str(1) + ".mat"
    exp_data = scipy.io.loadmat(filename)
    disp = exp_data["disp1"][::25].reshape(-1)
    force = exp_data["force1"][::25].reshape(-1)
    pass


def loss_curve():
    rnn_loss_path = "./dataset/csb/rnn.pkl"
    birnn_loss_path = "./dataset/csb/birnn.pkl"
    rnn_loss = torch.load(rnn_loss_path)
    birnn_loss = torch.load(birnn_loss_path)
    step = np.linspace(0, 30000, 301)
    step = step[1:]
    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 7 * cm))
    ax.plot(
        step,
        rnn_loss["train_loss_list"],
        color="blue",
        linewidth=1.2,
        label="RNN training",
    )
    ax.plot(
        step,
        rnn_loss["test_loss_list"],
        color="blue",
        linewidth=1.2,
        linestyle="--",
        label="RNN test",
    )
    ax.plot(
        step,
        birnn_loss["train_loss_list"],
        color="red",
        linewidth=1.2,
        label="BiRNN training",
    )

    ax.plot(
        step,
        birnn_loss["test_loss_list"],
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="BiRNN test",
    )
    ax.set_xlabel(r"Epoch ($\times$10$^4$)", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_xlim(0, 30000)
    ax.set_ylim(2e-4, 1e-2)
    ax.set_xticks(
        [0, 4000, 8000, 12000, 16000, 20000],
        ["0.0", "0.4", "0.8", "1.2", "1.6", "2.0"],
    )
    # ax.set_yticks([1e-4, 1e-3, 1e-2])
    ax.tick_params(axis="both", direction="in", which="both")
    ax.grid(True)

    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
    )
    plt.savefig("./figures/F_csb_rnn_training.pdf", bbox_inches="tight")
    plt.show()


def state_pred():
    rnn_state_pred, rnn_state_test = cb.test_rnn()
    birnn_state_pred, _ = cb.test_birnn()
    lw = 0.8
    step = 2000
    dof1 = 50
    dof2 = 75
    idx = 1
    time = np.arange(0, 0.001 * step, 0.001)
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 12 * cm))
    axs[0].plot(
        time, rnn_state_test[idx, :step, dof1], color="k", linewidth=lw, label="Ref."
    )
    axs[0].plot(
        time,
        rnn_state_pred[idx, :step, dof1],
        color="blue",
        linewidth=lw,
        label="RNN pred.",
        linestyle="-.",
    )
    axs[0].plot(
        time,
        birnn_state_pred[idx, :step, dof1],
        color="red",
        linewidth=lw,
        label="BiRNN pred.",
        linestyle="--",
    )
    axs[0].set_ylabel("Deflection (mm)")
    axs[0].set_ylim(-0.2, 0.2)
    fig.legend(
        bbox_to_anchor=(0.5, 0.95),
        loc="outside upper center",
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        ncol=3,
    )
    axs[0].text(
        -0.1 / 3, -0.1, "(a)", ha="center", va="center", transform=axs[0].transAxes
    )
    # x1, x2, y1, y2 = 1.1, 1.4, -0.1, 0.1
    # axins = axs[0].inset_axes(
    #     [0.73, 0.03, 0.26, 0.26],
    #     xlim=(x1, x2),
    #     ylim=(y1, y2),
    #     xticklabels=[],
    #     yticklabels=[],
    # )
    # axins.plot(
    #     time,
    #     rnn_state_test[0, :step, dof1],
    #     color="k",
    #     linewidth=1.2,
    # )
    # axins.plot(
    #     time,
    #     rnn_state_pred[0, :step, dof1],
    #     color="blue",
    #     linewidth=1.2,
    #     linestyle="-.",
    # )
    # axins.plot(
    #     time,
    #     birnn_state_pred[0, :step, dof1],
    #     color="red",
    #     linewidth=1.2,
    #     linestyle="--",
    # )
    # axins.set_xticks([])
    # axins.set_yticks([])
    # axs[0].indicate_inset_zoom(axins, edgecolor="black")

    axs[1].plot(
        time,
        rnn_state_test[idx, :step, dof2 + 128],
        color="k",
        linewidth=lw,
        label="Ref.",
    )
    axs[1].plot(
        time,
        rnn_state_pred[idx, :step, dof2 + 128],
        color="blue",
        linewidth=lw,
        label="RNN pred.",
        linestyle="-.",
    )
    axs[1].plot(
        time,
        birnn_state_pred[idx, :step, dof2 + 128],
        color="red",
        linewidth=lw,
        label="BiRNN pred.",
        linestyle="--",
    )
    axs[1].set_ylabel("Rotational speed (rad/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(-0.2, 0.2)
    axs[1].text(
        -0.1 / 3, -0.1, "(b)", ha="center", va="center", transform=axs[1].transAxes
    )
    for i in range(2):
        axs[i].tick_params(axis="both", direction="in", which="both")
        axs[i].grid(True)
        axs[i].set_xlim(0, 2)

    plt.savefig("./figures/F_csb_state_pred.pdf", bbox_inches="tight")
    plt.show()


def input_acc():
    # load the experimental data in .mat format
    data_length = 20000
    filename = f"./dataset/csb/exp_" + str(1) + ".mat"
    exp_data = scipy.io.loadmat(filename)
    acc1 = -exp_data["acc1"][:data_length:5].reshape(-1, 1)
    acc2 = -exp_data["acc2"][:data_length:5].reshape(-1, 1)
    acc3 = -exp_data["acc3"][:data_length:5].reshape(-1, 1)
    time = np.arange(0, 0.0002 * data_length, 0.001)
    fig, axs = plt.subplots(3, 1, figsize=(18 * cm, 12 * cm))
    axs[0].plot(time, acc3, color="black", linewidth=0.8)
    axs[0].set_ylim(-3.2, 2.2)
    axs[0].set_yticks([-3.2, -1.6, 0, 1.1, 2.2])
    axs[0].text(
        -0.2 * 2 / 9,
        -0.1,
        "(a)",
        ha="center",
        va="center",
        transform=axs[0].transAxes,
    )
    axs[0].text(
        0.2 * 2 / 9,
        0.2,
        "A1",
        ha="center",
        va="center",
        transform=axs[0].transAxes,
    )
    axs[1].plot(time, acc2, color="black", linewidth=0.8)
    axs[1].set_ylim(-2.6, 2.6)
    axs[1].set_yticks([-2.6, -1.3, 0, 1.3, 2.6])
    axs[1].text(
        -0.2 * 2 / 9,
        -0.1,
        "(b)",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].text(
        0.2 * 2 / 9,
        0.2,
        "A2",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[2].plot(time, acc1, color="black", linewidth=0.8)
    axs[2].set_ylim(-4.8, 3.2)
    axs[2].set_yticks([-4.8, -2.4, 0, 1.6, 3.2])
    axs[2].text(
        -0.2 * 2 / 9,
        -0.1,
        "(c)",
        ha="center",
        va="center",
        transform=axs[2].transAxes,
    )
    axs[2].text(
        0.2 * 2 / 9,
        0.2,
        "A3",
        ha="center",
        va="center",
        transform=axs[2].transAxes,
    )
    axs[1].set_ylabel("Acceleration (g)")
    axs[2].set_xlabel("Time (s)")
    for i in range(3):
        axs[i].set_xlim(0, 4)
        axs[i].tick_params(axis="both", direction="in", which="both")
        axs[i].grid(True)
    plt.savefig("./figures/F_csb_input_acc.pdf", bbox_inches="tight")
    plt.show()


def rnn_birnn_pred():
    rnn_pred, ground_truth = cb.rnn_pred()
    birnn_pred, _ = cb.birnn_pred()
    shift = 11
    data_length = 4000
    time = np.arange(0, 0.001 * data_length, 0.001)
    fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 6 * cm))
    ax.plot(
        time,
        ground_truth[shift : data_length + shift],
        color="black",
        linewidth=0.8,
        label="Ref.",
    )
    ax.plot(
        time,
        rnn_pred[:data_length],
        color="blue",
        linewidth=0.8,
        label="RNN pred.",
        linestyle="-.",
    )
    ax.plot(
        time,
        birnn_pred[:data_length],
        color="red",
        linewidth=0.8,
        label="BiRNN pred.",
        linestyle="--",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_ylim(-0.12, 0.12)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-0.12, -0.06, 0, 0.06, 0.12])
    ax.tick_params(axis="both", direction="in", which="both")
    ax.grid(True)
    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        loc="upper left",
        ncol=3,
    )
    plt.savefig("./figures/F_csb_rnn_birnn_pred.pdf", bbox_inches="tight")
    plt.show()
