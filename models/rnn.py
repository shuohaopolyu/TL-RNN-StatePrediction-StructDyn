import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define RNN
class Rnn(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=10,
        output_size=6,
        num_layers=1,
        bidirectional=False,
    ):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            # bias = False,
        ).to(device)
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, output_size).to(device)
        else:
            self.linear = nn.Linear(hidden_size, output_size).to(device)
        # the dropout layer is not used in the paper, because the input data and hidden state are not too large.
        # the weight decay (L2 regularization) is used in the paper to avoid overfitting.
        # but one can try to use the dropout layer to see if it can improve the performance.
        # self.dropout_1 = nn.Dropout(p=0.1)
        # self.dropout_2 = nn.Dropout(p=0.1)

    def forward(self, u, h0):
        # u = self.dropout_1(u)
        y, hn = self.rnn(u, h0)
        # y = self.dropout_2(y)
        y = self.linear(y)
        return y, hn

    def train_RNN(
        self,
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs,
        learning_rate,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=1e-6,
    ):
        assert test_set["X"].shape[0] % train_set["X"].shape[0] == 0, "Wrong data size"
        length_ratio = test_set["X"].shape[0] // train_set["X"].shape[0]
        input_length = train_set["X"].shape[0]
        # Define the loss function
        loss_fn = nn.MSELoss(reduction="mean")
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # Define the loss list
        train_loss_list = []
        test_loss_list = []
        # Start training
        for epoch in range(epochs):
            self.train()
            # Train the model
            output_train, _ = self.forward(train_set["X"], train_h0)
            loss_train = loss_fn(output_train, train_set["Y"])
            optimizer.zero_grad()
            loss_train.backward(retain_graph=True)
            optimizer.step()
            if (epoch + 1) % 2000 == 0:
                train_loss_list.append(loss_train.item())
                # Test the model
                self.eval()
                with torch.no_grad():
                    hn = test_h0
                    for i in range(length_ratio):
                        input_test_X = test_set["X"][
                            i * input_length : (i + 1) * input_length, :
                        ]
                        pred, hn = self.forward(input_test_X, hn)
                        if i == 0:
                            output_test = pred
                        else:
                            output_test = torch.cat((output_test, pred), dim=0)
                loss_test = loss_fn(output_test, test_set["Y"])
                test_loss_list.append(loss_test.item())
                if train_msg:
                    print(
                        "Epoch: %d / %d, Train loss: %.4e, Test loss: %.4e"
                        % (epoch + 1, epochs, loss_train.item(), loss_test.item())
                    )
                # Save the model
                if model_save_path is not None:
                    torch.save(self.state_dict(), model_save_path)
                    # print(f"Model saved to {model_save_path}")

                if loss_save_path is not None:
                    torch.save(
                        {
                            "train_loss_list": train_loss_list,
                            "test_loss_list": test_loss_list,
                        },
                        loss_save_path,
                    )
                    # print(f"Loss saved to {loss_save_path}")
        return train_loss_list, test_loss_list


# Define LSTM
class Lstm(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=10,
        output_size=6,
        num_layers=1,
        bidirectional=False,
    ):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            # bias=False,
            bidirectional=bidirectional,
            batch_first=True,
        ).to(device)
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, output_size, bias=True).to(device)
        else:
            self.linear = nn.Linear(hidden_size, output_size, bias=True).to(device)
        # self.dropout_1 = nn.Dropout(p=0.1)
        # self.dropout_2 = nn.Dropout(p=0.1)

    def forward(self, u, h0, c0):
        # u = self.dropout_1(u)
        y, (hn, cn) = self.lstm(u, (h0, c0))
        # y = self.dropout_2(y)
        y = self.linear(y)
        return y, hn, cn

    def train_LSTM(
        self,
        train_set,
        test_set,
        train_hc0,
        test_hc0,
        epochs,
        learning_rate,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=1e-6,
    ):
        assert test_set["X"].shape[0] % train_set["X"].shape[0] == 0, "Wrong data size"
        length_ratio = test_set["X"].shape[0] // train_set["X"].shape[0]
        input_length = train_set["X"].shape[0]
        # Define the loss function
        loss_fn = nn.MSELoss(reduction="mean")
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # Define the loss list
        train_loss_list = []
        test_loss_list = []
        # Start training
        for epoch in range(epochs):
            self.train()
            # Train the model
            output_train, _, _ = self.forward(
                train_set["X"], train_hc0[0], train_hc0[1]
            )
            loss_train = loss_fn(output_train, train_set["Y"])
            optimizer.zero_grad()
            loss_train.backward(retain_graph=True)
            optimizer.step()
            if (epoch + 1) % 2000 == 0:
                train_loss_list.append(loss_train.item())
                # Test the model
                self.eval()
                with torch.no_grad():
                    hn, cn = test_hc0
                    for i in range(length_ratio):
                        input_test_X = test_set["X"][
                            i * input_length : (i + 1) * input_length, :
                        ]
                        pred, hn, cn = self.forward(input_test_X, hn, cn)
                        if i == 0:
                            output_test = pred
                        else:
                            output_test = torch.cat((output_test, pred), dim=0)
                loss_test = loss_fn(output_test, test_set["Y"])
                test_loss_list.append(loss_test.item())
                if train_msg:
                    print(
                        "Epoch: %d / %d, Train loss: %.4e, Test loss: %.4e"
                        % (epoch + 1, epochs, loss_train.item(), loss_test.item())
                    )
                # Save the model
                if model_save_path is not None:
                    torch.save(self.state_dict(), model_save_path)
                    # print(f"Model saved to {model_save_path}")
                if loss_save_path is not None:
                    torch.save(
                        {
                            "train_loss_list": train_loss_list,
                            "test_loss_list": test_loss_list,
                        },
                        loss_save_path,
                    )
                    # print(f"Loss saved to {loss_save_path}")
        return train_loss_list, test_loss_list
