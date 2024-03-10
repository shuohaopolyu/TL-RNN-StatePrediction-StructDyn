import numpy as np
import matplotlib.pyplot as plt
import torch
from models import Rnn
from exps import shear_type_structure

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
    plt.grid(True)
    plt.savefig("./figures/loss_curve.svg", bbox_inches="tight")
    plt.show()


def disp_pred():
    file_idx = 2
    dof_idx = 2
    BiRNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    RNN4ststate = Rnn(
        input_size=5,
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    with open("./dataset/sts/birnn.pth", "rb") as f:
        BiRNN4ststate.load_state_dict(torch.load(f))
    RNN4ststate.eval()
    BiRNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 10
    rnn_test_h0 = torch.zeros(1, num_test_files, 30).to(device)
    birnn_test_h0 = torch.zeros(2, num_test_files, 30).to(device)
    _, _, state_test, acc_test = shear_type_structure.training_test_data(
        [0, 1, 2, 3, 4], 1, 40
    )
    with torch.no_grad():
        rnn_state_pred, _ = RNN4ststate(acc_test, rnn_test_h0)
        birnn_state_pred, _ = BiRNN4ststate(acc_test, birnn_test_h0)

    rnn_state_pred = rnn_state_pred.cpu().numpy()
    rnn_state_pred = rnn_state_pred[:, :, dof_idx]
    birnn_state_pred = birnn_state_pred.cpu().numpy()
    birnn_state_pred = birnn_state_pred[:, :, dof_idx]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, dof_idx]
    plt.plot(state_test[file_idx, :], label="Ground truth", color="k", linewidth=0.8)
    plt.plot(
        birnn_state_pred[file_idx, :],
        label="BiRNN prediction",
        linestyle="--",
        color="r",
        linewidth=0.8,
    )
    plt.plot(
        rnn_state_pred[file_idx, :],
        label="RNN prediction",
        linestyle="-.",
        color="b",
        linewidth=0.8,
    )
    plt.legend()
    plt.show()


def velo_pred():
    pass
