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

    plt.plot(
        np.arange(1, 15),
        error_1[1:, 2],
        label="Displacement: Train",
        marker="o",
        color="dimgrey",
    )
    plt.plot(
        np.arange(1, 15),
        error_2[1:, 2],
        label="Displacement: Test 1",
        marker="s",
        color="dimgrey",
        linestyle="dashed",
    )
    plt.plot(
        np.arange(1, 15),
        error_3[1:, 2],
        label="Displacement: Test 2",
        marker="^",
        color="dimgrey",
        linestyle="-.",
    )
    plt.plot(
        np.arange(1, 15),
        error_4[1:, 2],
        label="Velocity: Train",
        marker="D",
        color="darkred",
    )
    plt.plot(
        np.arange(1, 15),
        error_5[1:, 2],
        label="Velocity: Test 1",
        marker="v",
        color="darkred",
        linestyle="dashed",
    )
    plt.plot(
        np.arange(1, 15),
        error_6[1:, 2],
        label="Velocity: Test 2",
        marker="P",
        color="darkred",
        linestyle="-.",
    )
    plt.yscale("log")
    plt.xlim(1, 14)
    plt.ylim(1e-9, 1e-2)
    plt.tick_params(axis="both", direction="in", which="both", top=True, right=True)
    plt.xticks(np.arange(1, 15, 1))
    plt.legend()
    plt.xlabel("Number of Hidden Neurons")
    plt.ylabel("Mean Saure Error")
    plt.tight_layout()
    plt.savefig("./figures/num_hid_neuron.png", dpi=300)
    plt.show()


def data_size():
    # needs modification
    with open("./dataset/ae_train_data_size.pkl", "rb") as f:
        data = pickle.load(f)
    error_disp_train_mean = data["error_disp_train_mean"]
    error_disp_test_1_mean = data["error_disp_test_1_mean"]
    error_disp_test_2_mean = data["error_disp_test_2_mean"]
    error_velo_train_mean = data["error_velo_train_mean"]
    error_velo_test_1_mean = data["error_velo_test_1_mean"]
    error_velo_test_2_mean = data["error_velo_test_2_mean"]

    num_data = np.array([20, 50, 100, 200, 500, 1000, 2000, 4000, 8000])

    # fill the interval between max and min
    plt.plot(
        num_data,
        error_disp_train_mean[:, 2],
        marker="o",
        color="dimgrey",
        label="Displacement: Train",
    )
    plt.plot(
        num_data,
        error_disp_test_1_mean[:, 2],
        marker="s",
        color="dimgrey",
        linestyle="dashed",
        label="Displacement: Test 1",
    )
    plt.plot(
        num_data,
        error_disp_test_2_mean[:, 2],
        marker="^",
        color="dimgrey",
        linestyle="-.",
        label="Displacement: Test 2",
    )
    plt.plot(
        num_data,
        error_velo_train_mean[:, 2],
        marker="D",
        color="darkred",
        label="Velocity: Train",
    )
    plt.plot(
        num_data,
        error_velo_test_1_mean[:, 2],
        marker="v",
        color="darkred",
        linestyle="dashed",
        label="Velocity: Test 1",
    )
    plt.plot(
        num_data,
        error_velo_test_2_mean[:, 2],
        marker="P",
        color="darkred",
        linestyle="-.",
        label="Velocity: Test 2",
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-9, 1e-3)
    plt.xlim(20, 8000)
    plt.tick_params(axis="both", direction="in", which="both", top=True, right=True)
    plt.legend()
    plt.xlabel("Number of Training Data")
    plt.ylabel("Mean Saure Error")
    plt.tight_layout()
    plt.savefig("./figures/data_size.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # set the font family
    plt.rc("font", family="serif")
    plt.rc("font", size=12)
    num_hid_neuron()
    data_size()
