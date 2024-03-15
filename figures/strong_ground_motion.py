from systems import BaseIsolatedStructure
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from models import Rnn
from excitations import FlatNoisePSD, PSDExcitationGenerator
from scipy import interpolate
import os

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Arial"
# set the font size's default value
plt.rcParams.update({"font.size": 10})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def processNGAfile(filepath, scalefactor=None):
    try:
        if not scalefactor:
            scalefactor = 1.0
        with open(filepath, "r") as f:
            content = f.readlines()
        counter = 0
        desc, row4Val, acc_data = "", "", []
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0][0] == "N":
                    val = row4Val.split()
                    npts = float(val[(val.index("NPTS=")) + 1].rstrip(","))
                    dt = float(val[(val.index("DT=")) + 1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value) * scalefactor
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time = []
                for i in range(0, len(acc_data)):
                    t = i * dt
                    time.append(t)
            counter = counter + 1
        return desc, npts, dt, time, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")


def plot_ground_motion():
    acc_file_name_list = [
        "./dataset/bists/ath.KOBE.NIS000.AT2",
        "./dataset/bists/RSN12_KERN.PEL_PEL090.AT2",
        "./dataset/bists/RSN22_ELALAMO_ELC180.AT2",
        "./dataset/bists/RSN570_SMART1.45_45C00DN.AT2",
        # "./dataset/bists/RSN22_ELALAMO_ELC270.AT2",
        # "./dataset/bists/RSN286_ITALY_A-BIS000.AT2",
    ]
    xlim_list = [[0, 40], [0, 70], [0, 60], [0, 50], [0, 60], [0, 40]]
    ylim_list = [
        [-0.6, 0.6],
        [-0.1, 0.1],
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-0.4, 0.4],
        [-0.7, 0.7],
    ]
    yticks_list = [
        np.arange(-0.6, 0.7, 0.3),
        np.arange(-0.1, 0.11, 0.05),
        np.arange(-0.3, 0.4, 0.15),
        np.arange(-0.3, 0.4, 0.15),
        np.arange(-0.4, 0.5, 0.1),
        np.arange(-0.7, 0.8, 0.2),
    ]
    xticks_list = [
        np.arange(0, 41, 10),
        np.arange(0, 71, 10),
        np.arange(0, 61, 10),
        np.arange(0, 51, 10),
        np.arange(0, 61, 10),
        np.arange(0, 41, 10),
    ]

    text_desc = ["Kobe", "Kern County", "El Álamo", "Taiwan"]
    factors = [1, 1, 6, 2, 2, 2]
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 10 * cm))
    for i, acc_file_i in enumerate(acc_file_name_list):
        desc, _, _, time, inp_acc = processNGAfile(acc_file_i)
        print(desc)
        acc_g = inp_acc * factors[i]
        axs[i // 2, i % 2].plot(time, acc_g, "k", linewidth=0.8)
        axs[i // 2, i % 2].tick_params(
            which="both", direction="in", right=True, top=True
        )
        axs[i // 2, i % 2].grid(True)
        axs[i // 2, i % 2].set_xlim(xlim_list[i])
        axs[i // 2, i % 2].set_ylim(ylim_list[i])
        axs[i // 2, i % 2].set_yticks(yticks_list[i])
        axs[i // 2, i % 2].set_xticks(xticks_list[i])
        axs[i // 2, i % 2].text(
            0.8,
            0.85,
            text_desc[i],
            ha="center",
            va="center",
            transform=axs[i // 2, i % 2].transAxes,
        )

    fig.supxlabel("Time (s)")
    fig.supylabel("Acceleration (g)", position=(0.05, 0.5))
    plt.savefig("./figures/strong_ground_motion.svg", bbox_inches="tight")
    plt.show()
