import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    plt.plot(dkf_disp_1[num_dof, :600], label="DKF", color="dimgrey")
    plt.plot(birnn_ae_disp_1[:600, num_dof], label="BiRNN-AE", color="darkred")
    plt.plot(rnn_ae_disp_1[:600, num_dof], label="RNN-AE", color="darkorange")
    plt.plot(bilstm_ae_disp_1[:600, num_dof], label="BiLSTM-AE", color="darkgreen")
    plt.plot(lstm_ae_disp_1[:600, num_dof], label="LSTM-AE", color="darkblue")
    plt.plot(disp_1[:600, num_dof], label="Ground Truth", color="black")
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
    models_performance_eval()