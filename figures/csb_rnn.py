import numpy as np
import matplotlib.pyplot as plt
import pickle
from exps import continuous_beam as cb

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 8})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


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
    fig, axs = plt.subplots(3, 2, figsize=(18 * cm, 12 * cm))
    axs[0, 0].plot(freq, amp_data[:, 0], color="red")
    axs[0, 0].plot(freq, amp_mtx[:, 0], color="blue", linestyle="--")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel(
        r"Magnitude (m/s$^2$/N)",
    )
    axs[0, 1].plot(freq, phase_data[:, 0], color="red", linewidth=1.2)
    axs[0, 1].plot(freq, phase_mtx[:, 0], color="blue", linestyle="--", linewidth=1.2)
    axs[0, 1].set_ylabel("Phase (rad)")
    axs[1, 0].plot(freq, amp_data[:, 1], color="red", linewidth=1.2)
    axs[1, 0].plot(freq, amp_mtx[:, 1], color="blue", linestyle="--", linewidth=1.2)
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_ylabel(r"Magnitude (m/s$^2$/N)")
    axs[1, 1].plot(freq, phase_data[:, 1], color="red", linewidth=1.2)
    axs[1, 1].plot(freq, phase_mtx[:, 1], color="blue", linestyle="--", linewidth=1.2)
    axs[1, 1].set_ylabel("Phase (rad)")
    axs[2, 0].plot(freq, amp_data[:, 2], color="red", linewidth=1.2)
    axs[2, 0].plot(freq, amp_mtx[:, 2], color="blue", linestyle="--", linewidth=1.2)
    axs[2, 0].set_yscale("log")
    axs[2, 0].set_ylabel(r"Magnitude (m/s$^2$/N)")
    axs[2, 1].plot(freq, phase_data[:, 2], color="red", linewidth=1.2)
    axs[2, 1].plot(freq, phase_mtx[:, 2], color="blue", linestyle="--", linewidth=1.2)
    axs[2, 1].set_ylabel("Phase (rad)")
    for i in range(3):
        for j in range(2):
            axs[i, j].tick_params(axis="y", direction="in", which="both")
            axs[i, j].set_xlim(10, 100)
            axs[i, 1].set_ylim(0, 3.15)
            # axs[i, 0].set_ylim(1, 1000)
            axs[i, j].grid(True)
            axs[2, j].set_xlabel("Frequency (Hz)")
            axs[i, j].text(
                -0.06,
                -0.1,
                figidx[i * 2 + j],
                ha="center",
                va="center",
                transform=axs[i, j].transAxes,
            )
            axs[i, j].tick_params(axis="x", direction="in", which="both")
            axs[i, j].set_xticks([10, 20, 40, 60, 80, 100])
            axs[i, j].set_xticklabels(["10", "20", "40", "60", "80", "100"])
    plt.tight_layout()
    plt.savefig("./figures/F_csb_model_updating.pdf", bbox_inches="tight")
    plt.show()
