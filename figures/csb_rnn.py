import numpy as np
import matplotlib.pyplot as plt
import pickle
from exps import continuous_beam as cb
import scipy.io
import scipy.signal


# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 8})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def ema():
    data_path = "./dataset/csb/noise_for_ema_2.mat"
    exp_data = scipy.io.loadmat(data_path)
    acc1 = -exp_data["Data1_AI_1_AI_1"][2000:7000, 0].reshape(-1, 1) * 9.8
    acc2 = -exp_data["Data1_AI_2_AI_2"][2000:7000, 0].reshape(-1, 1) * 9.8
    acc3 = -exp_data["Data1_AI_3_AI_3"][2000:7000, 0].reshape(-1, 1) * 9.8
    force = exp_data["Data1_AI_4_AI_4"][2000:7000, 0].reshape(-1, 1)
    time = np.arange(0, 0.0002 * len(acc1), 0.0002)
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 8 * cm))
    axs[0, 0].plot(time, acc1, color="black", linewidth=0.6)
    axs[0, 0].set_ylabel("Acceleration (m/s$^2$)")
    axs[0, 0].set_xlim(0, 1)
    axs[0, 1].plot(time, acc2, color="black", linewidth=0.6)
    axs[0, 1].set_ylabel("Acceleration (m/s$^2$)")
    axs[0, 1].set_xlim(0, 1)
    axs[1, 0].plot(time, acc3, color="black", linewidth=0.6)
    axs[1, 0].set_ylabel("Acceleration (m/s$^2$)")
    axs[1, 0].set_xlim(0, 1)
    axs[1, 1].plot(time, force, color="black", linewidth=0.6)
    axs[1, 1].set_ylabel("Force (N)")
    axs[1, 1].set_xlim(0, 1)
    fig.supxlabel("Time (s)", fontsize=8)
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
    amp_data = np.abs(frf_data)
    amp_mtx = np.abs(frf_model)
    freq = np.linspace(10, 100, 450)
    figidx = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fig, axs = plt.subplots(2, 3, figsize=(18 * cm, 10 * cm))
    axs[0, 0].plot(freq, amp_data[:, 0], color="red", linewidth=1.2)
    axs[0, 0].plot(freq, amp_mtx[:, 0], color="blue", linestyle="--", linewidth=1.2)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel(
        r"Magnitude (m/s$^2$/N)",
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


def filteer_disp():
    filename = f"./dataset/csb/exp_" + str(1) + ".mat"
    exp_data = scipy.io.loadmat(filename)
    disp = exp_data["disp1"][::]
    # plt.plot(disp, color="black", linewidth=0.8)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Displacement (mm)")
    # plt.show()
    # power spectral density of displacement
    print(disp.shape)
    f, Pxx = scipy.signal.periodogram(disp.squeeze(), fs=5000, scaling="spectrum")
    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 8 * cm))
    print(Pxx)
    ax.plot(f, np.sqrt(Pxx), color="black", linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectrum")
    # ax.set_xlim(0, 25)
    ax.set_yscale("log")
    plt.show()
