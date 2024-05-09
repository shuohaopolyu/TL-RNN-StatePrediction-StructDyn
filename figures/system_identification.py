import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import numpy.linalg as LA
from matplotlib.legend import _get_legend_handles_labels
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 8})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def acceleration_measurement():
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    time = solution["time"]
    acc = solution["acc"]
    # four subfigures to plot the time history of the seismic response
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.plot(time, acc[0, :], label="1st floor")
    ax1.set_ylabel("Acceleration (m/s^2)")

    ax2.plot(time, acc[4, :], label="5nd floor")

    ax3.plot(time, acc[8, :], label="9rd floor")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s^2)")

    ax4.plot(time, acc[12, :], label="13th floor")
    ax4.set_xlabel("Time (s)")

    plt.show()


def psd_acc():
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    time = solution["time"]
    acc = solution["acc"]
    acc_test = np.squeeze(acc[12, :])
    N = len(acc_test)
    yf_test = np.fft.fft(acc_test)
    xf = np.linspace(0.0, 10, N // 2)
    # plot the PSD of the acceleration
    f, Pxx = signal.welch(acc, fs=20, window="hann", nperseg=2000, noverlap=500, axis=1)
    # plt.plot(xf, 2.0 * time[-1] * (np.abs(yf_test[: N // 2])**2)/N**2)
    plt.plot(f, Pxx.T)
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.show()


def fdd(signal_mtx, f_lb=0.3, f_ub=0.8, nperseg_num=1000, type="peak"):
    # implementation of frequency domain decomposition
    # signal should in matrix form, whose dimension is 5*n_t
    # will return the mode shapes and the natural frequency
    # two ways to generate mode shapes, peak or average
    w_f = []
    w_acc = []
    for i in range(signal_mtx.shape[0]):
        for j in range(signal_mtx.shape[0]):
            w_f_temp, w_acc_temp = signal.csd(
                signal_mtx[i, :],
                signal_mtx[j, :],
                fs=20,
                window="hann",
                nperseg=nperseg_num,
                axis=0,
                scaling="density",
                average="mean",
            )
            w_f.append(w_f_temp)
            w_acc.append(w_acc_temp)
    idx = [i for i, v in enumerate(w_f[0]) if v <= f_ub and v >= f_lb]
    tru_w_acc = np.array(w_acc)[:, idx]
    nf_temp_idx = []
    ms = []
    for i in range(tru_w_acc.shape[1]):
        G_yy = tru_w_acc[:, i].reshape(13, 13)
        u, s, _ = LA.svd(G_yy, full_matrices=True)
        nf_temp_idx.append(s[0])
        ms.append(np.real(u[:, 0]))
    nf_temp_idx = np.argmax(np.array(nf_temp_idx))
    nf_idx = idx[0] + nf_temp_idx
    nf = w_f[0][nf_idx]
    if type == "peak":
        ms_peak = np.array(ms)[nf_temp_idx, :]
        return ms_peak, nf
    elif type == "average":
        ms_avg = np.average(np.array(ms), axis=0)
        return ms_avg, nf


def ms_acc(f_lb=1.1, f_ub=1.5):
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    acc_mtx = solution["acc"]
    ms, nf = fdd(acc_mtx, f_lb=f_lb, f_ub=f_ub, nperseg_num=1000, type="peak")
    plt.plot(range(13), ms, "-o")
    plt.xlabel("DOF")
    plt.ylabel("Mode Shape")


def vib_hist():
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data = data["acc"].T
    time = np.arange(0, data.shape[0] / 20, 1 / 20)
    fig, ax = plt.subplots(3, 1, figsize=(8 * cm, 8 * cm))
    plt.subplot(3, 1, 1)
    plt.plot(time[:200], data[:200, 0], label="1st floor")
    plt.subplot(3, 1, 2)
    plt.plot(time[:200], data[:200, 6], label="7th floor")
    plt.ylabel("Acceleration (m/s^2)")
    plt.subplot(3, 1, 3)
    plt.plot(time[:200], data[:200, 12], label="13th floor")
    plt.xlabel("Time (s)")
    plt.show()


def base_loads():
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    disp = data["disp"].T
    z = data["z"].T
    time = np.arange(0, disp.shape[0] / 20, 1 / 20)
    alpha = 0.7
    base_elastic_force = alpha * 1300 * disp[:, 0]
    base_hysteretic_force = (1 - alpha) * 10 * z
    plt.figure(figsize=(18 * cm, 6 * cm))
    plt.plot(
        time[:2000],
        base_elastic_force[:2000] * 1.2e2,
        label="Elastic force",
        color="red",
        linewidth=0.8,
    )
    plt.plot(
        time[:2000],
        base_hysteretic_force[:2000] * 1.2e2,
        label="Hysteretic force",
        color="blue",
        linewidth=0.8,
    )
    plt.xlim(0, 100)
    plt.ylim(-10, 10)
    plt.yticks(np.arange(-10, 11, 5), fontsize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Base shear force (kN)")
    plt.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
    )
    plt.grid(True)
    plt.tick_params(axis="both", direction="in")
    plt.savefig("./figures/base_loads.svg", bbox_inches="tight")
    plt.savefig("./figures/F_base_loads.pdf", bbox_inches="tight")
    plt.show()


def singular_values():
    from pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
    from pyoma2.OMA import SingleSetup

    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data = data["acc"].T
    Pali_ss = SingleSetup(data, fs=20)
    fsdd = FSDD_algo(name="FSDD", nxseg=2000, method_SD="per", pov=0.5)
    Pali_ss.add_algorithms(fsdd)
    Pali_ss.run_by_name("FSDD")
    fig, ax = fsdd.plot_CMIF(freqlim=(0.2, 8))

    fig.set_size_inches(9 * cm, 7 * cm)
    plt.tick_params(axis="both", direction="in")
    plt.ylim([-80, 10])
    plt.ylabel("Singular values of \n cross-spectral matrices (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("")
    plt.savefig("./figures/singular_values.svg", bbox_inches="tight")
    plt.savefig("./figures/F_singular_values.pdf", bbox_inches="tight")
    plt.show()


def mode_shape_old():
    # load the seismic response
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    model_ms = solution["model_ms"]
    ms = solution["ms"]
    fig, axs = plt.subplots(1, 5, figsize=(10 * cm, 14 * cm))
    for i in range(5):
        ms_i = ms[:, i] / np.max(np.abs(ms[:, i]))
        model_ms_i = model_ms[:, i] / np.max(np.abs(model_ms[:, i]))
        if ms_i[0] * model_ms_i[0] < 0:
            model_ms_i = -model_ms_i
        axs[i].plot(
            model_ms_i,
            range(13),
            "-s",
            color="red",
            markersize=6,
            linewidth=1,
            label="Model results",
            markerfacecolor="None",
        )
        axs[i].plot(
            ms_i,
            range(13),
            "--o",
            color="blue",
            markersize=6,
            linewidth=1,
            label="Measurements",
            markerfacecolor="None",
        )
        print(model_ms_i.T @ ms_i / (LA.norm(model_ms_i) * LA.norm(ms_i)))
        axs[i].tick_params(axis="both", direction="in")
        axs[i].set_ylim(-0.5, 12.5)
        axs[i].set_xlim(-1.1, 1.1)
        axs[i].grid(True)
        axs[i].set_yticks(range(13), [str(i + 1) for i in range(13)], fontsize=8)
        axs[i].set_xticks([0], [str(i + 1)], fontsize=8)
        # axs[i].set_xlabel(idx_list[i])
        if i == 0:
            axs[i].set_ylabel("Degree of freedom")
            axs[0].legend(
                loc="upper center",
                bbox_to_anchor=(3, 1.1),
                fontsize=8,
                facecolor="white",
                edgecolor="black",
                ncol=2,
            )
        else:
            axs[i].set_yticklabels([])
    axs[2].set_xlabel("Mode")
    # plt.savefig("./figures/mode_shape.svg", bbox_inches="tight")
    # plt.show()


def mode_shape(i=0):
    # load the seismic response
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    model_ms = solution["model_ms"]
    ms = solution["ms"]
    # fig, axs = plt.subplots(1, 5, figsize=(10 * cm, 14 * cm))
    ms_i = ms[:, i] / np.max(np.abs(ms[:, i]))
    model_ms_i = model_ms[:, i] / np.max(np.abs(model_ms[:, i]))
    if ms_i[0] * model_ms_i[0] < 0:
        model_ms_i = -model_ms_i
    plt.plot(
        model_ms_i,
        range(13),
        "-s",
        color="red",
        markersize=6,
        linewidth=1,
        label="Model results",
        markerfacecolor="None",
    )
    plt.plot(
        ms_i,
        range(13),
        "--o",
        color="blue",
        markersize=6,
        linewidth=1,
        label="Measurements",
        markerfacecolor="None",
    )
    # print(model_ms_i.T @ ms_i / (LA.norm(model_ms_i) * LA.norm(ms_i)))
    plt.tick_params(axis="both", direction="in")
    plt.ylim(-0.5, 12.5)
    plt.xlim(-1.1, 1.1)
    plt.grid(True)
    plt.yticks(range(13), [str(i + 1) for i in range(13)], fontsize=8)
    plt.xticks([0], [str(i + 1)], fontsize=8)
    # axs[i].set_xlabel(idx_list[i])
    if i == 0:
        plt.ylabel("Degree of freedom", labelpad=0)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(3, 1.06),
            fontsize=8,
            facecolor="white",
            edgecolor="black",
            ncol=2,
        )
        plt.text(
            -0.1,
            -0.1 * 0.4,
            "(c)",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        plt.yticks(alpha=0)
        plt.ylabel(None)
    if i == 2:
        plt.xlabel("Mode")


def natural_frequency():
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    model_nf = solution["model_nf"]
    nf = solution["nf"]
    # fig, ax = plt.subplots(1, 1, figsize=(8 * cm, 6 * cm))
    x = np.arange(5)
    plt.bar(
        x - 0.125 + 1,
        model_nf[:5],
        0.25,
        color="red",
        label="Model results",
    )
    plt.bar(x + 0.125 + 1, nf[:5], 0.25, color="blue", label="Measurements")
    plt.xlabel("Mode", fontsize=8, labelpad=0)
    plt.tick_params(axis="both", direction="in")
    plt.ylabel("Natural frequency (Hz)", labelpad=1)
    plt.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
    )
    plt.text(-0.1, -0.1, "(b)", ha="center", va="center", transform=plt.gca().transAxes)
    # plt.savefig("./figures/natural_frequency.svg", bbox_inches="tight")
    # print((model_nf[:5] - nf[:5]) / nf[:5])
    # plt.show()


def params():

    # Set the LaTeX preamble to use specific packages or font settings
    # plt.rcParams["text.latex.preamble"] = [
    #     r"\usepackage{amsmath}",  # for mathematical expressions
    #     r"\usepackage{amssymb}",  # for mathematical symbols
    #     r"\usepackage{mathpazo}",  # use the Palatino font
    # ]
    k_sub = [
        r"$k_1$",
        r"$k_2$",
        r"$k_3$",
        r"$k_4$",
        r"$k_5$",
        r"$k_6$",
        r"$k_7$",
        r"$k_8$",
        r"$k_9$",
        r"$k_{10}$",
        r"$k_{11}$",
        r"$k_{12}$",
        r"$k_{13}$",
    ]
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    measured_params = solution["params"]
    model_params = np.array([13, 12, 12, 12, 8, 8, 8, 8, 8, 5, 5, 5, 5]) * 0.12
    measured_params = model_params * measured_params
    model_params[0] = 0
    x = np.arange(13, dtype=float)
    x[0] = 0.15

    # fig, ax = plt.subplots(1, 1, figsize=(8 * cm, 8 * cm))
    plt.barh(
        x - 0.15 + 1, measured_params, 0.3, label="Updated parameters", color="blue"
    )
    plt.barh(x + 0.15 + 1, model_params, 0.3, label="True parameters", color="red")
    plt.yticks(range(1, 14, 1))
    plt.tick_params(axis="both", direction="in")
    plt.legend(
        fontsize=8,
        facecolor="white",
        edgecolor="black",
    )
    plt.xlim(0, 1.6)
    plt.xticks(np.arange(0, 1.7, 0.4))
    plt.xlabel(r"Stiffness parameter values ($\times$10$^5$ kN/m)", labelpad=1)
    plt.yticks(x + 1, k_sub, fontsize=8)
    plt.text(-0.1, -0.1, "(a)", ha="center", va="center", transform=plt.gca().transAxes)

    # plt.savefig("./figures/params.svg", bbox_inches="tight")
    # plt.show()


def model_updating():
    # combine the above theree figures
    gs = gridspec.GridSpec(15, 19)
    pl.figure(figsize=(19 * cm, 15 * cm))
    ax1 = pl.subplot(gs[:8, :8])
    params()
    ax2 = pl.subplot(gs[9:, :8])
    natural_frequency()
    ax3 = pl.subplot(gs[:, 9:11])
    mode_shape(0)
    ax4 = pl.subplot(gs[:, 11:13])
    mode_shape(1)
    ax5 = pl.subplot(gs[:, 13:15])
    mode_shape(2)
    ax6 = pl.subplot(gs[:, 15:17])
    mode_shape(3)
    ax7 = pl.subplot(gs[:, 17:])
    mode_shape(4)

    plt.savefig("./figures/F_model_updating.pdf", bbox_inches="tight")
