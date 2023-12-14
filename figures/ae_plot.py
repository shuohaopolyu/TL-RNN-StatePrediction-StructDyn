import numpy as np
import pickle
import matplotlib.pyplot as plt


def num_hid_neuron():

    with open("./dataset/ae_num_hid_neuron.pkl", "rb") as f:
        data = pickle.load(f)

    error_1 = data["error_1"]
    error_2 = data["error_2"]
    error_3 = data["error_3"]
    error_4 = data["error_4"]

    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, 15), error_1[:, 2], label="disp_1", marker="o", color="blue")
    ax1.plot(np.arange(1, 15), error_2[:, 2], label="disp_2", marker="s", color="blue", linestyle="dashed")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, 15), error_3[:, 2], label="velo_1", marker="^", color="red")
    ax2.plot(np.arange(1, 15), error_4[:, 2], label="velo_2", marker="D", color="red", linestyle="dashed")
    ax2.set_yscale("log")

    plt.legend()
    plt.xlabel("Number of neurons in hidden layer")
    plt.ylabel("Mean Saure Error")
    plt.savefig("./figures/num_hid_neuron.png")
    plt.show()

with open("./dataset/ae_num_hid_neuron.pkl", "rb") as f:
    data = pickle.load(f)

error_1 = data["error_1"]
error_2 = data["error_2"]
error_3 = data["error_3"]
error_4 = data["error_4"]

plt.plot(np.arange(1, 15), error_1[:, 2], label="disp_1", marker="o", color="blue")
plt.plot(np.arange(1, 15), error_2[:, 2], label="disp_2", marker="s", color="blue", linestyle="dashed")
plt.plot(np.arange(1, 15), error_3[:, 2], label="velo_1", marker="^", color="red")
plt.plot(np.arange(1, 15), error_4[:, 2], label="velo_2", marker="D", color="red", linestyle="dashed")
plt.yscale("log")

plt.legend()
plt.xlabel("Number of neurons in hidden layer")
plt.ylabel("Mean Saure Error")
plt.savefig("./figures/num_hid_neuron.png")
plt.show()
