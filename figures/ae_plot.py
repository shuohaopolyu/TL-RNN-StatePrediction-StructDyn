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
    # needs modification
    with open("./dataset/ae_train_data_size.pkl", "rb") as f:
        data = pickle.load(f)
    error_disp_train_mean = data["error_disp_train_mean"]
    error_disp_train_std = data["error_disp_train_std"]
    error_disp_test_1_mean = data["error_disp_test_1_mean"]
    error_disp_test_1_std = data["error_disp_test_1_std"]
    error_disp_test_2_mean = data["error_disp_test_2_mean"]
    error_disp_test_2_std = data["error_disp_test_2_std"]
    error_velo_train_mean = data["error_velo_train_mean"]
    error_velo_train_std = data["error_velo_train_std"]
    error_velo_test_1_mean = data["error_velo_test_1_mean"]
    error_velo_test_1_std = data["error_velo_test_1_std"]
    error_velo_test_2_mean = data["error_velo_test_2_mean"]
    error_velo_test_2_std = data["error_velo_test_2_std"]

    num_data = np.array([20, 50, 100, 200, 500, 1000, 2000, 4000, 8000])


    # fill the interval between mean - std and mean + std
    plt.fill_between(num_data, error_disp_train_mean - error_disp_train_std, error_disp_train_mean + error_disp_train_std, alpha=0.2, color="blue")
    plt.fill_between(num_data, error_disp_test_1_mean - error_disp_test_1_std, error_disp_test_1_mean + error_disp_test_1_std, alpha=0.2, color="blue")
    plt.fill_between(num_data, error_disp_test_2_mean - error_disp_test_2_std, error_disp_test_2_mean + error_disp_test_2_std, alpha=0.2, color="blue")
    plt.fill_between(num_data, error_velo_train_mean - error_velo_train_std, error_velo_train_mean + error_velo_train_std, alpha=0.2, color="red")
    plt.fill_between(num_data, error_velo_test_1_mean - error_velo_test_1_std, error_velo_test_1_mean + error_velo_test_1_std, alpha=0.2, color="red")
    plt.fill_between(num_data, error_velo_test_2_mean - error_velo_test_2_std, error_velo_test_2_mean + error_velo_test_2_std, alpha=0.2, color="red")

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Number of training data")
    plt.ylabel("Mean Saure Error")
    plt.savefig("./figures/data_size.png")
    plt.show()

if __name__ == "__main__":
    # num_hid_neuron()
    data_size()