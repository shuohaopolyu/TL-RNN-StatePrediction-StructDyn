import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from excitations import FlatNoisePSD, PSDExcitationGenerator

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 7})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def compound_envelope(b1, b2, gamma, t_array):
    tmax = t_array[-1]
    # Assuming t_array is a numpy array
    envelope = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        normalized_time = t / tmax
        if normalized_time < b1:
            envelope[i] = t / (b1 * tmax)
        elif normalized_time > b2:
            envelope[i] = np.exp(-gamma * (normalized_time - b2))
        else:
            envelope[i] = 1
    return envelope


def cwt_acc_g():
    # compute the seismic vibration response
    psd_func = FlatNoisePSD(a_v=0.3)
    excitation = PSDExcitationGenerator(psd_func, 40, 10)
    time, acc_g = excitation.generate()
    b1 = np.random.uniform(0.1, 0.2)
    gamma = np.random.uniform(3, 5)
    b2 = np.random.uniform(0.4, 0.6)
    window_point = compound_envelope(b1, b2, gamma, time)
    print(np.std(acc_g))
    acc_g = acc_g * window_point

    widths = np.arange(1, 201)
    cwtmatr = signal.cwt(acc_g, signal.ricker, widths)
    cwtmatr_yflip = np.flipud(cwtmatr)

    fig, axs = plt.subplots(1, 2, figsize=(18 * cm, 7 * cm))
    axs[0].plot(time, acc_g / 9.8, color="b", linewidth=1.5)
    axs[0].tick_params(which="both", direction="in")
    axs[0].set_xlabel("Time (s)", fontsize=7)
    axs[0].set_ylabel("Acceleration (g)", color="b", fontsize=7)
    axs[0].set_xlim(0, 40)
    axs[0].set_xticks(
        np.arange(0, 41, 5),
        ["0", "5", "10", "15", "20", "25", "30", "35", "40"],
        fontsize=7,
    )
    axs[0].set_ylim(-0.8, 0.8)
    axs[0].set_yticks(
        np.arange(-0.8, 0.9, 0.2),
        ["-0.8", "-0.6", "-0.4", "-0.2", "0", "0.2", "0.4", "0.6", "0.8"],
    )
    axs[0].tick_params(axis="y", labelcolor="b", direction="in", which="both")
    axs[0].text(-4, -0.96, "(a)", fontsize=7)
    ax2 = axs[0].twinx()
    ax2.plot(time, window_point, color="r", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Window function", color="r", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="r", direction="in", which="both")
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks(
        np.arange(-1.2, 1.3, 0.4),
        ["-1.2", "-0.8", "-0.4", "0", "0.4", "0.8", "1.2"],
        fontsize=7,
    )
    axs[0].grid(True)
    # axs[0].legend(["Ground acceleration"], fontsize=10, facecolor="white", edgecolor="black")
    # ax2.legend(["Window function"], fontsize=10, facecolor="white", edgecolor="black")
    wv = axs[1].imshow(
        abs(cwtmatr_yflip),
        extent=[0, 40, 1 / 10, 10],
        cmap="seismic",
        aspect="auto",
        vmax=abs(cwtmatr).max(),
        vmin=0,
    )
    # axs[1].set_yscale('log')
    axs[1].set_xlabel("Time (s)", fontsize=7)
    axs[1].set_ylabel("Frequency (Hz)", fontsize=7)
    axs[1].tick_params(which="both", direction="in")
    axs[1].text(-4, -1, "(b)", fontsize=7)
    axs[1].set_yticks(
        [0.1, 2, 4, 6, 8, 10], ["0", "2", "4", "6", "8", "10"], fontsize=7
    )
    axs[1].set_xticks(np.arange(0, 41, 10), ["0", "10", "20", "30", "40"], fontsize=7)
    cbar = fig.colorbar(wv, ax=axs[1], orientation="vertical", extend="both")
    cbar.ax.tick_params(direction="in")

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(7)
    # cbar.minorticks_on()
    fig.tight_layout(pad=0.1)
    plt.savefig("./figures/cwt_acc_g.svg")
    plt.savefig("./figures/F_cwt_acc_g.pdf")
    plt.show()
