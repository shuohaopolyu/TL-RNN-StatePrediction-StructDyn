import numpy as np
import pickle
import matplotlib.pyplot as plt


def num_hid_neuron():

    with open("./dataset/ae_num_hid_neuron.pkl", "rb") as f:
        data = pickle.load(f)

    error_1 = data["error_disp_train"]
    error_2 = data["error_disp_test_1"]
    error_3 = data["error_disp_test_2"]
    error_4 = data["error_velo_train"]
    error_5 = data["error_velo_test_1"]
    error_6 = data["error_velo_test_2"]
    print(error_1[:, 2])

    plt.plot(np.arange(1, 15), error_1[1:, 2], label="error_disp_train", marker="o", color="blue")
    plt.plot(np.arange(1, 15), error_2[1:, 2], label="error_disp_test_1", marker="s", color="blue", linestyle="dashed")
    plt.plot(np.arange(1, 15), error_3[1:, 2], label="error_disp_test_2", marker="^", color="blue", linestyle="-.")
    plt.plot(np.arange(1, 15), error_4[1:, 2], label="error_velo_train", marker="D", color="red")
    plt.plot(np.arange(1, 15), error_5[1:, 2], label="error_velo_test_1", marker="v", color="red", linestyle="dashed")
    plt.plot(np.arange(1, 15), error_6[1:, 2], label="error_velo_test_2", marker="P", color="red", linestyle="-.")
    plt.yscale("log")

    plt.legend()
    plt.xlabel("Number of neurons in hidden layer")
    plt.ylabel("Mean Saure Error")
    plt.savefig("./figures/num_hid_neuron.png")
    plt.show()

def data_size():
    with open("./dataset/ae_train_data_size.pkl", "rb") as f:
        data = pickle.load(f)

    error_1 = data["error_1"]
    error_2 = data["error_2"]
    error_3 = data["error_3"]
    error_4 = data["error_4"]
    error_5 = data["error_5"]
    error_6 = data["error_6"]
    num_data = np.array([100, 200, 500, 1000, 2000, 4000, 8000])

    plt.plot(num_data, error_1[:, 2], label="disp_ext_1_test", marker="o", color="blue")
    plt.plot(num_data, error_2[:, 2], label="disp_ext_2", marker="s", color="blue", linestyle="dashed")
    plt.plot(num_data, error_3[:, 2], label="velo_ext_1_train", marker="^", color="blue", linestyle="-.")

    plt.plot(num_data, error_4[:, 2], label="velo_ext_1_test", marker="D", color="red")
    plt.plot(num_data, error_5[:, 2], label="velo_ext_2", marker="v", color="red", linestyle="dashed")
    plt.plot(num_data, error_6[:, 2], label="velo_ext_1_train", marker="P", color="red", linestyle="-.")

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Number of training data")
    plt.ylabel("Mean Saure Error")
    plt.savefig("./figures/data_size.png")
    plt.show()

if __name__ == "__main__":
    num_hid_neuron()
    # data_size()