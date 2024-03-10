import numpy as np
import matplotlib.pyplot as plt
import torch

# set the fonttype to be Arial
plt.rcParams["font.family"] = "Arial"
# set the font size's default value
plt.rcParams.update({"font.size": 10})
ts = {"fontname": "Times New Roman"}
cm = 1 / 2.54  # centimeters in inches


def loss_curve():
    loss_save_path = "./dataset/sts/birnn.pkl"
    with open(loss_save_path, "rb") as f:
        birnn = torch.load(f)
    loss_save_path = "./dataset/sts/rnn.pkl"
    with open(loss_save_path, "rb") as f:
        rnn = torch.load(f)
    epoch = np.arange(1, (len(birnn["train_loss_list"])) * 200, 200)
    plt.figure(figsize=(10 * cm, 8 * cm))
    plt.plot(epoch, birnn["train_loss_list"], color="r", linewidth=1.5)
    plt.plot(epoch, birnn["test_loss_list"], color="r", linestyle="--", linewidth=1.5)
    plt.plot(epoch, rnn["train_loss_list"], color="b", linewidth=1.5)
    plt.plot(epoch, rnn["test_loss_list"], color="b", linestyle="--", linewidth=1.5)
    plt.tick_params(which="both", direction="in")
    plt.legend(
        ["BiRNN training", "BiRNN test", "RNN training", "RNN test"],
        fontsize=10,
        facecolor="white",
        edgecolor="black",
    )
    plt.xlim(0, 50000)
    plt.xticks([0, 10000, 20000, 30000, 40000, 50000], ["0", "1", "2", "3", "4", "5"])
    plt.yscale("log")
    plt.xlabel(r"Epoch ($\times 10^4$)")
    plt.ylabel("Loss")
    plt.savefig("./figures/loss_curve.svg", bbox_inches="tight")
    plt.show()
