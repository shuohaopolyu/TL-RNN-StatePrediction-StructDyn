from systems import BaseIsolatedStructure
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from models import Rnn
from excitations import FlatNoisePSD, PSDExcitationGenerator
from scipy import interpolate
import os


"""
@author: Daniel Hutabarat - UC Berkeley, 2017
"""


def processNGAfile(filepath, scalefactor=None):
    """
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and
    time iterval of the recording.
    Parameters:
    ------------
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.

    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.

    Example: (plot time vs acceleration)
    filepath = os.path.join(os.getcwd(),'motion_1')
    desc, npts, dt, time, inp_acc = processNGAfile (filepath)
    plt.plot(time,inp_acc)

    """
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
        "./dataset/bists/RSN137_TABAS_BAJ-L1.AT2",
        "./dataset/bists/RSN570_SMART1.45_45C00DN.AT2",
        "./dataset/bists/RSN826_CAPEMEND_EUR000.AT2",
        "./dataset/bists/RSN832_LANDERS_ABY000.AT2",
    ]
    factors = [1, 6, 8, 8, 2, 5]
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    for i, acc_file_i in enumerate(acc_file_name_list):
        _, _, _, time, inp_acc = processNGAfile(acc_file_i)
        acc_g = inp_acc * 9.81 * factors[i]
        axs[i // 2, i % 2].plot(time, acc_g)
    plt.show()