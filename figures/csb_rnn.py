import numpy as np
import matplotlib.pyplot as plt
import pickle
from experiments import continuous_beam as cb
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
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 1].set_xlabel("Time (s)")

    # fig.supxlabel("Time (s)", fontsize=8)
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
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 2].set_xlabel("Frequency (Hz)")

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
    print(rnn_loss["train_loss_list"][-1])
    print(birnn_loss["train_loss_list"][-1])
    step1 = np.linspace(0, 50000, 501)
    step2 = np.linspace(0, 50000, 501)
    step1 = step1[1:]
    step2 = step2[1:]
    print(len(rnn_loss["train_loss_list"]))
    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 7 * cm))
    ax.plot(
        step1,
        rnn_loss["train_loss_list"][0:500],
        color="blue",
        linewidth=1.2,
        label="RNN training",
    )
    ax.plot(
        step1,
        rnn_loss["test_loss_list"][0:500],
        color="blue",
        linewidth=1.2,
        linestyle="--",
        label="RNN test",
    )
    ax.plot(
        step2,
        birnn_loss["train_loss_list"],
        color="red",
        linewidth=1.2,
        label="BiRNN training",
    )

    ax.plot(
        step2,
        birnn_loss["test_loss_list"],
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="BiRNN test",
    )
    ax.set_xlabel(r"Epoch ($\times$10$^4$)", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_xlim(0, 50000)
    ax.set_ylim(1e-4, 1e-1)
    ax.set_xticks(
        [0, 10000, 20000, 30000, 40000, 50000],
        ["0", "1", "2", "3", "4", "5"],
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
    start = 2000
    step = 2000
    dof1 = 50
    dof2 = 75
    idx = 0
    cmap = "seismic"
    time = np.arange(0, 0.0002 * step, 0.0002)
    fig, axs = plt.subplots(3, 2, figsize=(18 * cm, 12 * cm))
    wv = axs[0, 0].imshow(
        rnn_state_test[idx, start : step + start, 0:128:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="spline16",
    )

    axs[0, 0].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[0, 0].set_ylim(0, 63)
    axs[0, 0].tick_params(axis="both", direction="in", which="both")
    axs[0, 0].set_xlim(0, 2000)
    axs[0, 0].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[0, 0].set_ylabel("x", style="italic", labelpad=3)
    axs[0, 0].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[0, 0], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[0, 0].text(
        -0.1, -0.16, "(a)", ha="center", va="center", transform=axs[0, 0].transAxes
    )
    axs[0, 0].title.set_text("Ref. displacement field")
    wv = axs[1, 0].imshow(
        rnn_state_pred[idx, start : step + start, 0:128:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="spline16",
    )
    axs[1, 0].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[1, 0].set_ylim(0, 63)
    axs[1, 0].tick_params(axis="both", direction="in", which="both")
    axs[1, 0].set_xlim(0, 2000)
    axs[1, 0].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[1, 0].set_ylabel("x", style="italic", labelpad=3)
    axs[1, 0].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[1, 0], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[1, 0].text(
        -0.1, -0.16, "(b)", ha="center", va="center", transform=axs[1, 0].transAxes
    )
    axs[1, 0].title.set_text("RNN pred. displacement field")

    wv = axs[2, 0].imshow(
        birnn_state_pred[idx, start : step + start, 0:128:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="spline16",
    )
    axs[2, 0].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[2, 0].set_ylim(0, 63)
    axs[2, 0].tick_params(axis="both", direction="in", which="both")
    axs[2, 0].set_xlim(0, 2000)
    axs[2, 0].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[2, 0].set_ylabel("x", style="italic", labelpad=3)
    axs[2, 0].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[2, 0], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[2, 0].text(
        -0.1, -0.16, "(c)", ha="center", va="center", transform=axs[2, 0].transAxes
    )
    axs[2, 0].title.set_text("BiRNN pred. displacement field")

    wv = axs[0, 1].imshow(
        rnn_state_test[idx, start : step + start, 128:256:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )
    axs[0, 1].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[0, 1].set_ylim(0, 63)
    axs[0, 1].tick_params(axis="both", direction="in", which="both")
    axs[0, 1].set_xlim(0, 2000)
    axs[0, 1].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[0, 1].set_ylabel("x", style="italic", labelpad=3)
    axs[0, 1].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[0, 1], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[0, 1].text(
        -0.1, -0.16, "(d)", ha="center", va="center", transform=axs[0, 1].transAxes
    )
    axs[0, 1].title.set_text("Ref. velocity field")

    wv = axs[1, 1].imshow(
        rnn_state_pred[idx, start : step + start, 128:256:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )
    axs[1, 1].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[1, 1].set_ylim(0, 63)
    axs[1, 1].tick_params(axis="both", direction="in", which="both")
    axs[1, 1].set_xlim(0, 2000)
    axs[1, 1].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[1, 1].set_ylabel("x", style="italic", labelpad=3)
    axs[1, 1].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[1, 1], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[1, 1].text(
        -0.1, -0.16, "(e)", ha="center", va="center", transform=axs[1, 1].transAxes
    )
    axs[1, 1].title.set_text("RNN pred. velocity field")

    wv = axs[2, 1].imshow(
        birnn_state_pred[idx, start : step + start, 128:256:2].T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )
    axs[2, 1].set_yticks([0, 21, 42, 63], ["1.26", "0.84", "0.42", "0.00"])
    axs[2, 1].set_ylim(0, 63)
    axs[2, 1].tick_params(axis="both", direction="in", which="both")
    axs[2, 1].set_xlim(0, 2000)
    axs[2, 1].set_xticks(
        [0, 500, 1000, 1500, 2000], ["0.0", "0.1", "0.2", "0.3", "0.4"]
    )
    axs[2, 1].set_ylabel("x", style="italic", labelpad=3)
    axs[2, 1].set_xlabel("t", style="italic")
    cbar = fig.colorbar(wv, ax=axs[2, 1], orientation="vertical", extend="both")
    cbar.minorticks_on()
    axs[2, 1].text(
        -0.1, -0.16, "(f)", ha="center", va="center", transform=axs[2, 1].transAxes
    )
    axs[2, 1].title.set_text("BiRNN pred. velocity field")

    plt.tight_layout()
    plt.savefig("./figures/F_csb_state_img.pdf", bbox_inches="tight")
    plt.show()

    start = 0
    step = 10000
    time = np.arange(0, 0.0002 * step, 0.0002)

    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 12 * cm))
    axs[0].plot(
        time,
        rnn_state_test[idx, start : step + start, dof1],
        color="k",
        linewidth=lw,
        label="Ref.",
    )
    axs[0].plot(
        time,
        rnn_state_pred[idx, start : step + start, dof1],
        color="blue",
        linewidth=lw,
        label="RNN pred.",
        linestyle="-.",
    )
    axs[0].plot(
        time,
        birnn_state_pred[idx, start : step + start, dof1],
        color="red",
        linewidth=lw,
        label="BiRNN pred.",
        linestyle="--",
    )
    axs[0].set_ylabel("Deflection (mm)")
    axs[0].set_ylim(-1.0, 1.0)
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
    #     rnn_state_test[0, start:step+start, dof1],
    #     color="k",
    #     linewidth=1.2,
    # )
    # axins.plot(
    #     time,
    #     rnn_state_pred[0, start:step+start, dof1],
    #     color="blue",
    #     linewidth=1.2,
    #     linestyle="-.",
    # )
    # axins.plot(
    #     time,
    #     birnn_state_pred[0, start:step+start, dof1],
    #     color="red",
    #     linewidth=1.2,
    #     linestyle="--",
    # )
    # axins.set_xticks([])
    # axins.set_yticks([])
    # axs[0].indicate_inset_zoom(axins, edgecolor="black")

    axs[1].plot(
        time,
        rnn_state_test[idx, start : step + start, dof2 + 128],
        color="k",
        linewidth=lw,
        label="Ref.",
    )
    axs[1].plot(
        time,
        rnn_state_pred[idx, start : step + start, dof2 + 128],
        color="blue",
        linewidth=lw,
        label="RNN pred.",
        linestyle="-.",
    )
    axs[1].plot(
        time,
        birnn_state_pred[idx, start : step + start, dof2 + 128],
        color="red",
        linewidth=lw,
        label="BiRNN pred.",
        linestyle="--",
    )
    axs[1].set_ylabel("Rotational speed (rad/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(-0.6, 0.6)
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
    acc1 = -exp_data["acc1"][:data_length:1].reshape(-1, 1)
    acc2 = -exp_data["acc2"][:data_length:1].reshape(-1, 1)
    acc3 = -exp_data["acc3"][:data_length:1].reshape(-1, 1)
    time = np.arange(0, 0.0002 * data_length, 0.0002)
    fig, axs = plt.subplots(1, 3, figsize=(18 * cm, 6 * cm))
    axs[0].plot(time, acc3, color="black", linewidth=0.8)
    axs[0].set_ylim(-3.6, 3.6)
    axs[0].set_yticks([-3.6, -1.8, 0, 1.8, 3.6])
    axs[0].text(
        -0.1,
        -0.1,
        "(a)",
        ha="center",
        va="center",
        transform=axs[0].transAxes,
    )
    axs[0].text(
        0.1,
        0.1,
        "A1",
        ha="center",
        va="center",
        transform=axs[0].transAxes,
    )
    axs[0].set_xticks([0, 1, 2, 3, 4])
    axs[1].plot(time, acc2, color="black", linewidth=0.8)
    axs[1].set_ylim(-4.0, 4.0)
    axs[1].set_yticks([-4.0, -2.0, 0, 2.0, 4.0], ["-4.0", "-2.0", "0.0", "2.0", "4.0"])
    axs[1].text(
        -0.1,
        -0.1,
        "(b)",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].text(
        0.1,
        0.1,
        "A2",
        ha="center",
        va="center",
        transform=axs[1].transAxes,
    )
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[2].plot(time, acc1, color="black", linewidth=0.8)
    axs[2].set_ylim(-4.8, 4.8)
    axs[2].set_yticks([-4.8, -2.4, 0, 2.4, 4.8])
    axs[2].text(
        -0.1,
        -0.1,
        "(c)",
        ha="center",
        va="center",
        transform=axs[2].transAxes,
    )
    axs[2].text(
        0.1,
        0.1,
        "A3",
        ha="center",
        va="center",
        transform=axs[2].transAxes,
    )
    axs[2].set_xticks([0, 1, 2, 3, 4])
    axs[0].set_ylabel("Acceleration (g)")
    axs[2].set_xlabel("Time (s)")
    for i in range(3):
        axs[i].set_xlim(0, 4)
        axs[i].tick_params(axis="both", direction="in", which="both")
        axs[i].grid(True)
        axs[i].set_xlabel("Time (s)")
    plt.savefig("./figures/F_csb_input_acc.pdf", bbox_inches="tight")
    plt.show()


def rnn_birnn_pred():
    rnn_pred, ground_truth = cb.rnn_pred()
    birnn_pred, _ = cb.birnn_pred()
    shift = 55
    data_length = 20000
    time = np.arange(0, 0.0002 * data_length, 0.0002)
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
    ax.set_ylim(-0.16, 0.16)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-0.16, -0.08, 0, 0.08, 0.16])
    ax.tick_params(axis="both", direction="in", which="both")
    ax.grid(True)
    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        loc="lower left",
        ncol=3,
    )
    plt.savefig("./figures/F_csb_rnn_birnn_pred.pdf", bbox_inches="tight")
    plt.show()


def tr_loss_curve():
    dir1 = "./dataset/csb/tr_rnn.pkl"
    dir2 = "./dataset/csb/tr_birnn.pkl"
    rnn_loss = torch.load(dir1)
    birnn_loss = torch.load(dir2)
    steps = np.linspace(
        0,
        len(rnn_loss["train_loss_list"]),
        len(rnn_loss["train_loss_list"]) + 1,
    )
    steps = steps[1:]
    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 7 * cm))
    ax.plot(
        steps,
        rnn_loss["train_loss_list"],
        color="blue",
        linewidth=1.2,
        label="RNN training",
    )
    ax.plot(
        steps,
        rnn_loss["test_loss_list"],
        color="blue",
        linewidth=1.2,
        linestyle="--",
        label="RNN test",
    )
    steps = np.linspace(0, 198, 199)
    steps = steps[1:]
    ax.plot(
        steps,
        birnn_loss["train_loss_list"],
        color="red",
        linewidth=1.2,
        label="BiRNN training",
    )
    ax.plot(
        steps,
        birnn_loss["test_loss_list"],
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="BiRNN test",
    )
    ax.set_xlim(0, 250)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_ylim(5, 40)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", direction="in", which="both")
    ax.grid(True)
    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
    )
    plt.savefig("./figures/F_csb_tr_rnn_training.pdf", bbox_inches="tight")
    # ax.set_yscale("log")
    plt.show()


def tr_rnn_birnn_pred():
    rnn_disp_pred_filtered, disp = cb.rnn_pred(
        path="./dataset/csb/tr_rnn.pth", plot_data=False
    )

    birnn_disp_pred_filtered, _ = cb.birnn_pred(
        path="./dataset/csb/tr_birnn.pth", plot_data=False
    )
    shift = 52
    data_length = 20000
    time = np.arange(0, 0.0002 * data_length, 0.0002)
    fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 6 * cm))
    ax.plot(
        time,
        disp[shift : data_length + shift],
        color="black",
        linewidth=0.8,
        label="Ref.",
    )
    ax.plot(
        time,
        rnn_disp_pred_filtered[:data_length],
        color="blue",
        linewidth=0.8,
        label="TL-RNN pred.",
        linestyle="-.",
    )
    ax.plot(
        time,
        birnn_disp_pred_filtered[:data_length],
        color="red",
        linewidth=0.8,
        label="TL-BiRNN pred.",
        linestyle="--",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_ylim(-0.16, 0.16)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-0.16, -0.08, 0, 0.08, 0.16])
    ax.tick_params(axis="both", direction="in", which="both")
    ax.grid(True)
    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        loc="lower left",
        ncol=3,
    )
    plt.savefig("./figures/F_csb_tr_rnn_birnn_pred.pdf", bbox_inches="tight")
    plt.show()


def performance_evaluation():
    rnn_disp, disp = cb.rnn_pred(path="./dataset/csb/rnn.pth", plot_data=False)
    tr_rnn_disp, _ = cb.rnn_pred(path="./dataset/csb/tr_rnn.pth", plot_data=False)
    birnn_disp, _ = cb.birnn_pred(path="./dataset/csb/birnn.pth", plot_data=False)
    tr_birnn_disp, _ = cb.birnn_pred(path="./dataset/csb/tr_birnn.pth", plot_data=False)
    shift = 52
    start = 0
    data_length = 20000
    rnn_disp = rnn_disp[start:data_length].reshape(-1)
    tr_rnn_disp = tr_rnn_disp[start:data_length].reshape(-1)
    birnn_disp = birnn_disp[start:data_length].reshape(-1)
    tr_birnn_disp = tr_birnn_disp[start:data_length].reshape(-1)
    disp = disp[shift + start : data_length + shift].reshape(-1)

    # plt.plot(rnn_disp, label="RNN")
    # plt.plot(tr_rnn_disp, label="TL-RNN")
    # plt.plot(birnn_disp, label="BiRNN")
    # plt.plot(tr_birnn_disp, label="TL-BiRNN")
    # plt.plot(disp, label="Ref.", color="red")
    # plt.show()
    # calculate the error
    mean_target = np.mean(disp)
    metric = []
    metric.append(
        np.round(
            np.linalg.norm(disp - rnn_disp) / np.linalg.norm(disp - mean_target), 3
        )
    )
    metric.append(
        np.round(
            np.linalg.norm(disp - tr_rnn_disp) / np.linalg.norm(disp - mean_target), 3
        )
    )
    metric.append(
        np.round(
            np.linalg.norm(disp - birnn_disp) / np.linalg.norm(disp - mean_target), 3
        )
    )
    metric.append(
        np.round(
            np.linalg.norm(disp - tr_birnn_disp) / np.linalg.norm(disp - mean_target), 3
        )
    )
    print(np.linalg.norm(disp - rnn_disp) / np.linalg.norm(disp - mean_target))
    print(np.linalg.norm(disp - tr_rnn_disp) / np.linalg.norm(disp - mean_target))
    print(np.linalg.norm(disp - birnn_disp) / np.linalg.norm(disp - mean_target))
    print(np.linalg.norm(disp - tr_birnn_disp) / np.linalg.norm(disp - mean_target))

    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 7 * cm))
    rects = ax.bar(
        0,
        metric[0],
        width,
        label="Before TL",
        color="blue",
        zorder=3,
    )
    ax.bar_label(rects, padding=3)
    rects = ax.bar(
        0 + width,
        metric[1],
        width,
        label="After TL",
        color="red",
        zorder=3,
    )
    ax.bar_label(rects, padding=3)
    rects = ax.bar(
        1,
        metric[2],
        width,
        color="blue",
        zorder=3,
    )
    ax.bar_label(rects, padding=3)
    rects = ax.bar(
        1 + width,
        metric[3],
        width,
        color="red",
        zorder=3,
    )
    ax.bar_label(rects, padding=3, fmt="%.3f")
    ax.grid(True)
    ax.set_xticks([0.175, 0.5, 0.825, 1.175], ["RNN", "", "", "BiRNN"])
    ax.tick_params(axis="both", direction="in", which="both")
    ax.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        loc="upper right",
    )

    ax.set_ylabel("NRMSE")
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis="both", direction="in", which="both")
    plt.savefig("./figures/F_csb_performance_evaluation.pdf", bbox_inches="tight")
    plt.show()
