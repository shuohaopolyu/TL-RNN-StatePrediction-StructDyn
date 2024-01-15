import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch

def disp_loss_plot():

    # birnn_ae_disp
    with open("./dataset/birnn_ae_disp_loss.pkl", "rb") as f:
        birnn_ae_disp_loss = torch.load(f)
        brinn_ae_disp_train_loss = birnn_ae_disp_loss["train_loss_list"]
        brinn_ae_disp_test_loss = birnn_ae_disp_loss["test_loss_list"]

    # rnn_ae_disp
    with open("./dataset/rnn_ae_disp_loss.pkl", "rb") as f:
        rnn_ae_disp_loss = torch.load(f)
        rnn_ae_disp_train_loss = rnn_ae_disp_loss["train_loss_list"]
        rnn_ae_disp_test_loss = rnn_ae_disp_loss["test_loss_list"]

    # bilstm_ae_disp
    with open("./dataset/bilstm_ae_disp_loss.pkl", "rb") as f:
        bilstm_ae_disp_loss = torch.load(f)
        bilstm_ae_disp_train_loss = bilstm_ae_disp_loss["train_loss_list"]
        bilstm_ae_disp_test_loss = bilstm_ae_disp_loss["test_loss_list"]

    # lstm_ae_disp
    with open("./dataset/lstm_ae_disp_loss.pkl", "rb") as f:
        lstm_ae_disp_loss = torch.load(f)
        lstm_ae_disp_train_loss = lstm_ae_disp_loss["train_loss_list"]
        lstm_ae_disp_test_loss = lstm_ae_disp_loss["test_loss_list"]

    epochs = np.arange(0, 100000, 2000)
    epoch_plot = epochs[1:len(brinn_ae_disp_train_loss)+1]
    plt.plot(epoch_plot, brinn_ae_disp_train_loss, label="BiRNN-AE Train", color="darkred")
    plt.plot(epoch_plot, brinn_ae_disp_test_loss, label="BiRNN-AE Test", color="darkred", linestyle="--")
    epoch_plot = epochs[1:len(rnn_ae_disp_train_loss)+1]
    plt.plot(epoch_plot, rnn_ae_disp_train_loss, label="RNN-AE Train", color="darkorange")
    plt.plot(epoch_plot, rnn_ae_disp_test_loss, label="RNN-AE Test", color="darkorange", linestyle="--")
    epoch_plot = epochs[1:len(bilstm_ae_disp_train_loss)+1]
    plt.plot(epoch_plot, bilstm_ae_disp_train_loss, label="BiLSTM-AE Train", color="darkgreen")
    plt.plot(epoch_plot, bilstm_ae_disp_test_loss, label="BiLSTM-AE Test", color="darkgreen", linestyle="--")
    epoch_plot = epochs[1:len(lstm_ae_disp_train_loss)+1]
    plt.plot(epoch_plot, lstm_ae_disp_train_loss, label="LSTM-AE Train", color="darkblue")
    plt.plot(epoch_plot, lstm_ae_disp_test_loss, label="LSTM-AE Test", color="darkblue", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    # plt.xticks(np.arange(0, 100000, 10000))
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/disp_loss_plot.png", dpi=300)
    plt.show()

def velo_loss_plot():
    
    # birnn_ae_velo
    with open("./dataset/birnn_ae_velo_loss.pkl", "rb") as f:
        birnn_ae_velo_loss = torch.load(f)
        brinn_ae_velo_train_loss = birnn_ae_velo_loss["train_loss_list"]
        brinn_ae_velo_test_loss = birnn_ae_velo_loss["test_loss_list"]

    # rnn_ae_velo
    with open("./dataset/rnn_ae_velo_loss.pkl", "rb") as f:
        rnn_ae_velo_loss = torch.load(f)
        rnn_ae_velo_train_loss = rnn_ae_velo_loss["train_loss_list"]
        rnn_ae_velo_test_loss = rnn_ae_velo_loss["test_loss_list"]

    # bilstm_ae_velo
    with open("./dataset/bilstm_ae_velo_loss.pkl", "rb") as f:
        bilstm_ae_velo_loss = torch.load(f)
        bilstm_ae_velo_train_loss = bilstm_ae_velo_loss["train_loss_list"]
        bilstm_ae_velo_test_loss = bilstm_ae_velo_loss["test_loss_list"]

    # lstm_ae_velo
    with open("./dataset/lstm_ae_velo_loss.pkl", "rb") as f:
        lstm_ae_velo_loss = torch.load(f)
        lstm_ae_velo_train_loss = lstm_ae_velo_loss["train_loss_list"]
        lstm_ae_velo_test_loss = lstm_ae_velo_loss["test_loss_list"]

    epochs = np.arange(0, 100001, 2000)
    epoch_plot = epochs[1:len(brinn_ae_velo_train_loss)+1]
    plt.plot(epoch_plot, brinn_ae_velo_train_loss, label="BiRNN-AE Train", color="darkred")
    plt.plot(epoch_plot, brinn_ae_velo_test_loss, label="BiRNN-AE Test", color="darkred", linestyle="--")
    epoch_plot = epochs[1:len(rnn_ae_velo_train_loss)+1]
    plt.plot(epoch_plot, rnn_ae_velo_train_loss, label="RNN-AE Train", color="darkorange")
    plt.plot(epoch_plot, rnn_ae_velo_test_loss, label="RNN-AE Test", color="darkorange", linestyle="--")
    epoch_plot = epochs[1:len(bilstm_ae_velo_train_loss)+1]
    plt.plot(epoch_plot, bilstm_ae_velo_train_loss, label="BiLSTM-AE Train", color="darkgreen")
    plt.plot(epoch_plot, bilstm_ae_velo_test_loss, label="BiLSTM-AE Test", color="darkgreen", linestyle="--")
    epoch_plot = epochs[1:len(lstm_ae_velo_train_loss)+1]
    plt.plot(epoch_plot, lstm_ae_velo_train_loss, label="LSTM-AE Train", color="darkblue")
    plt.plot(epoch_plot, lstm_ae_velo_test_loss, label="LSTM-AE Test", color="darkblue", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    # plt.xticks(np.arange(0, 30000, 10000))
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/velo_loss_plot.png", dpi=300)
    plt.show()



def models_performance_eval():
    # dual kalman filter
    with open("./dataset/dkf_eval.pkl", "rb") as f:
        data = pickle.load(f)
    dkf_disp_1 = data["dkf_disp_1"]
    dkf_disp_2 = data["dkf_disp_2"]
    dkf_velo_1 = data["dkf_velo_1"]
    dkf_velo_2 = data["dkf_velo_2"]

    # birnn_ae
    with open("./dataset/birnn_ae_disp_pred_1.pkl", "rb") as f:
        birnn_ae_disp_1 = pickle.load(f)
    with open("./dataset/birnn_ae_disp_pred_2.pkl", "rb") as f:
        birnn_ae_disp_2 = pickle.load(f)
    with open("./dataset/birnn_ae_velo_pred_1.pkl", "rb") as f:
        birnn_ae_velo_1 = pickle.load(f)
    with open("./dataset/birnn_ae_velo_pred_2.pkl", "rb") as f:
        birnn_ae_velo_2 = pickle.load(f)

    # rnn_ae
    with open("./dataset/rnn_ae_disp_pred_1.pkl", "rb") as f:
        rnn_ae_disp_1 = pickle.load(f)
    with open("./dataset/rnn_ae_disp_pred_2.pkl", "rb") as f:
        rnn_ae_disp_2 = pickle.load(f)
    with open("./dataset/rnn_ae_velo_pred_1.pkl", "rb") as f:
        rnn_ae_velo_1 = pickle.load(f)
    with open("./dataset/rnn_ae_velo_pred_2.pkl", "rb") as f:
        rnn_ae_velo_2 = pickle.load(f)

    # bilstm_ae
    with open("./dataset/bilstm_ae_disp_pred_1.pkl", "rb") as f:
        bilstm_ae_disp_1 = pickle.load(f)
    with open("./dataset/bilstm_ae_disp_pred_2.pkl", "rb") as f:
        bilstm_ae_disp_2 = pickle.load(f)
    with open("./dataset/bilstm_ae_velo_pred_1.pkl", "rb") as f:
        bilstm_ae_velo_1 = pickle.load(f)
    with open("./dataset/bilstm_ae_velo_pred_2.pkl", "rb") as f:
        bilstm_ae_velo_2 = pickle.load(f)

    # lstm_ae
    with open("./dataset/lstm_ae_disp_pred_1.pkl", "rb") as f:
        lstm_ae_disp_1 = pickle.load(f)
    with open("./dataset/lstm_ae_disp_pred_2.pkl", "rb") as f:
        lstm_ae_disp_2 = pickle.load(f)
    with open("./dataset/lstm_ae_velo_pred_1.pkl", "rb") as f:
        lstm_ae_velo_1 = pickle.load(f)
    with open("./dataset/lstm_ae_velo_pred_2.pkl", "rb") as f:
        lstm_ae_velo_2 = pickle.load(f)

    # ground truth
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data = pickle.load(f)
    disp_1 = full_data["displacement"][2000:, :]
    velo_1 = full_data["velocity"][2000:, :]
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data = pickle.load(f)
    disp_2 = full_data["displacement"]
    velo_2 = full_data["velocity"]

    num_dof = 40
    num_data = 4000
    # plt.plot(dkf_disp_1[num_dof, :num_data-1] - np.mean(dkf_disp_1[num_dof, :num_data-1]), label="DKF", color="dimgrey")
    # plt.plot(birnn_ae_disp_1[1:num_data, num_dof], label="BiRNN-AE", color="darkred")
    # plt.plot(rnn_ae_disp_1[1:num_data, num_dof], label="RNN-AE", color="darkorange")
    # plt.plot(bilstm_ae_disp_1[1:num_data, num_dof], label="BiLSTM-AE", color="darkgreen")
    # plt.plot(lstm_ae_disp_1[1:num_data, num_dof], label="LSTM-AE", color="darkblue")
    # plt.plot(disp_1[1:num_data, num_dof], label="Ground Truth", color="black")
    # plt.legend()

    plt.plot(dkf_velo_1[num_dof, :num_data], label="DKF", color="dimgrey")
    plt.plot(birnn_ae_velo_1[:num_data, num_dof], label="BiRNN-AE", color="darkred")
    plt.plot(rnn_ae_velo_1[:num_data, num_dof], label="RNN-AE", color="darkorange")
    plt.plot(bilstm_ae_velo_1[:num_data, num_dof], label="BiLSTM-AE", color="darkgreen")
    plt.plot(lstm_ae_velo_1[:num_data, num_dof], label="LSTM-AE", color="darkblue")
    plt.plot(velo_1[:num_data, num_dof], label="Ground Truth", color="black")
    plt.legend()

    # # plot
    # fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    # ax[0, 0].plot(dkf_disp_1, label="DKF", color="dimgrey")
    # ax[0, 0].plot(birnn_ae_disp_1, label="BiRNN-AE", color="darkred")
    # ax[0, 0].plot(rnn_ae_disp_1, label="RNN-AE", color="darkorange")
    # ax[0, 0].plot(bilstm_ae_disp_1, label="BiLSTM-AE", color="darkgreen")
    # ax[0, 0].plot(lstm_ae_disp_1, label="LSTM-AE", color="darkblue")
    # ax[0, 0].plot(disp_1, label="Ground Truth", color="black")
    # ax[0, 0].set_xlabel("Time Step")
    # ax[0, 0].set_ylabel("Displacement")
    # ax[0, 0].legend()
    # ax[0, 0].set_title("Excitation Pattern 1")

    # ax[0, 1].plot(dkf_disp_2, label="DKF", color="dimgrey")
    # ax[0, 1].plot(birnn_ae_disp_2, label="BiRNN-AE", color="darkred")
    # ax[0, 1].plot(rnn_ae_disp_2, label="RNN-AE", color="darkorange")
    # ax[0, 1].plot(bilstm_ae_disp_2, label="BiLSTM-AE", color="darkgreen")
    # ax[0, 1].plot(lstm_ae_disp_2, label="LSTM-AE", color="darkblue")
    # ax[0, 1].plot(disp_2, label="Ground Truth", color="black")
    # ax[0, 1].set_xlabel("Time Step")
    # ax[0, 1].set_ylabel("Displacement")
    # ax[0, 1].legend()
    # ax[0, 1].set_title("Excitation Pattern 2")

    # ax[1, 0].plot(dkf_velo_1, label="DKF", color="dimgrey")
    # ax[1, 0].plot(birnn_ae_velo_1, label="BiRNN-AE", color="darkred")
    # ax[1, 0].plot(rnn_ae_velo_1, label="RNN-AE", color="darkorange")
    # ax[1, 0].plot(bilstm_ae_velo_1, label="BiLSTM-AE", color="darkgreen")
    # ax[1, 0].plot(lstm_ae_velo_1, label="LSTM-AE", color="darkblue")
    # ax[1, 0].plot(velo_1, label="Ground Truth", color="black")
    # ax[1, 0].set_xlabel("Time Step")
    # ax[1, 0].set_ylabel("Velocity")
    # ax[1, 0].legend()
    # ax[1, 0].set_title("Excitation Pattern 1")

    # ax[1, 1].plot(dkf_velo_2, label="DKF", color="dimgrey")
    # ax[1, 1].plot(birnn_ae_velo_2, label="BiRNN-AE", color="darkred")
    # ax[1, 1].plot(rnn_ae_velo_2, label="RNN-AE", color="darkorange")
    # ax[1, 1].plot(bilstm_ae_velo_2, label="BiLSTM-AE", color="darkgreen")
    # ax[1, 1].plot(lstm_ae_velo_2, label="LSTM-AE", color="darkblue")
    # ax[1, 1].plot(velo_2, label="Ground Truth", color="black")
    # ax[1, 1].set_xlabel("Time Step")
    # ax[1, 1].set_ylabel("Velocity")
    # ax[1, 1].legend()
    # ax[1, 1].set_title("Excitation Pattern 2")

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # set the font family
    plt.rc("font", family="serif")
    plt.rc("font", size=12)
    # models_performance_eval()
    # disp_loss_plot()
    velo_loss_plot()