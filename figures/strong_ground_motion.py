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
plt.rcParams["font.family"] = "Times New Roman"
# set the font size's default value
plt.rcParams.update({"font.size": 7})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches
plt.rcParams["axes.unicode_minus"] = True


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
    yticks_str_list = [
        ["\N{MINUS SIGN}0.6", "\N{MINUS SIGN}0.3", "0", "0.3", "0.6"],
        ["\N{MINUS SIGN}0.1", "\N{MINUS SIGN}0.05", "0", "0.05", "0.1"],
        ["\N{MINUS SIGN}0.3", "\N{MINUS SIGN}0.15", "0", "0.15", "0.3"],
        ["\N{MINUS SIGN}0.3", "\N{MINUS SIGN}0.15", "0", "0.15", "0.3"],
        [
            "\N{MINUS SIGN}0.4",
            "\N{MINUS SIGN}0.3",
            "\N{MINUS SIGN}0.2",
            "\N{MINUS SIGN}0.1",
            "0",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
        ],
        [
            "\N{MINUS SIGN}0.7",
            "\N{MINUS SIGN}0.5",
            "\N{MINUS SIGN}0.3",
            "\N{MINUS SIGN}0.1",
            "0",
            "0.1",
            "0.3",
            "0.5",
            "0.7",
        ],
    ]
    xticks_list = [
        np.arange(0, 41, 10),
        np.arange(0, 71, 10),
        np.arange(0, 61, 10),
        np.arange(0, 51, 10),
        np.arange(0, 61, 10),
        np.arange(0, 41, 10),
    ]
    xticks_str_list = [
        ["0", "10", "20", "30", "40"],
        ["0", "10", "20", "30", "40", "50", "60", "70"],
        ["0", "10", "20", "30", "40", "50", "60"],
        ["0", "10", "20", "30", "40", "50"],
        ["0", "10", "20", "30", "40", "50", "60"],
        ["0", "10", "20", "30", "40"],
    ]

    text_desc = ["Kobe, 1995", "Kern County, 1952", "El Álamo, 1956", "Taiwan, 1986"]
    factors = [1, 1, 6, 2, 2, 2]
    fig, axs = plt.subplots(2, 2, figsize=(16.4 * cm, 7 * cm))
    for i, acc_file_i in enumerate(acc_file_name_list):
        desc, _, _, time, inp_acc = processNGAfile(acc_file_i)
        print(desc)
        acc_g = inp_acc * factors[i]
        axs[i // 2, i % 2].plot(time, acc_g, "k", linewidth=0.8)
        axs[i // 2, i % 2].tick_params(
            which="both", direction="in", right=False, top=False
        )
        axs[i // 2, i % 2].grid(True)
        axs[i // 2, i % 2].set_xlim(xlim_list[i])
        axs[i // 2, i % 2].set_ylim(ylim_list[i])
        axs[i // 2, i % 2].set_yticks(yticks_list[i], yticks_str_list[i], fontsize=7)
        axs[i // 2, i % 2].set_xticks(xticks_list[i], xticks_str_list[i], fontsize=7)
        axs[i // 2, i % 2].text(
            0.5,
            0.85,
            text_desc[i],
            ha="center",
            va="center",
            transform=axs[i // 2, i % 2].transAxes,
            fontsize=7,
        )
        if i > 1:
            axs[i // 2, i % 2].set_xlabel("Time (s)", fontsize=7)
    axs[0, 0].set_ylabel("Acceleration (g)", fontsize=7)
    axs[1, 0].set_ylabel("Acceleration (g)", fontsize=7)

    # fig.supxlabel("Time (s)", fontsize=7)
    # fig.supylabel("Acceleration (g)", position=(0.05, 0.5), fontsize=7)
    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/strong_ground_motion.svg")
    plt.savefig("./figures/F_strong_ground_motion.pdf")
    plt.show()
