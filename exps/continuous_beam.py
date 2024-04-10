import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt


def random_vibration():
    psd = BandPassPSD(a_v=2.25, f_1=10.0, f_2=410.0)
    force = PSDExcitationGenerator(psd, tmax=10, fmax=500)
    force = force()
    cb = ContinuousBeam01(t_eval=np.linspace(0, 10 - 1 / 1000, 10000), f_t=[force])
    full_data = cb.response(type="full", method="Radau")
    solution = {}
    solution["displacement"] = full_data["displacement"].T
    solution["acceleration"] = full_data["acceleration"].T
    solution["velocity"] = full_data["velocity"].T
    solution["force"] = cb.f_mtx().T
    solution["time"] = cb.t_eval.reshape(-1, 1)
    file_name = f"./dataset/csb/solution" + format(0, "03") + ".pkl"
    with open(file_name, "wb") as f:
        pickle.dump(solution, f)
    print("File " + file_name + " saved.")


def plot_solution():
    with open("./dataset/csb/solution000.pkl", "rb") as f:
        solution = pickle.load(f)
    plt.plot(solution["time"], solution["displacement"][:, 0])
    plt.plot(solution["time"], solution["displacement"][:, 36])
    plt.show()
