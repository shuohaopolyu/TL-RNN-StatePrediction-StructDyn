import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            bias=False,
        ).to(device)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, output_size, bias=False).to(device)
        else:
            self.linear = nn.Linear(hidden_size, output_size, bias=False).to(device)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(output_size, output_size, bias=False).to(device)
        self.linear3 = nn.Linear(output_size, output_size, bias=False).to(device)
        self.linear4 = nn.Linear(output_size, output_size, bias=False).to(device)
        self.linear5 = nn.Linear(output_size, output_size, bias=False).to(device)

    def forward(self, u, h0):
        y, hn = self.rnn(u, h0)
        y = self.linear(y)
        y = self.tanh(y)
        y = self.linear2(y)
        y = self.tanh(y)
        y = self.linear3(y)
        y = self.tanh(y)
        y = self.linear4(y)
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
        weight_decay=0.0,
    ):
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
            if (epoch + 1) % 200 == 0:
                train_loss_list.append(loss_train.item())
                # Test the model
                self.eval()
                with torch.no_grad():
                    hn = test_h0
                    output_test, _ = self.forward(test_set["X"], hn)
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
                    print(f"Model saved to {model_save_path}")

                if loss_save_path is not None:
                    torch.save(
                        {
                            "train_loss_list": train_loss_list,
                            "test_loss_list": test_loss_list,
                        },
                        loss_save_path,
                    )
                    print(f"Loss saved to {loss_save_path}")
        return train_loss_list, test_loss_list


class Rnn02(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=10,
        output_size=6,
        num_layers=1,
        bidirectional=False,
    ):
        super(Rnn02, self).__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            bias=False,
        ).to(device)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False).to(
                device
            )
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(2 * hidden_size, output_size, bias=False).to(
                device
            )
            self.linear3 = nn.Linear(output_size, output_size, bias=False).to(device)
        else:
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(hidden_size, output_size, bias=False).to(device)
            self.linear3 = nn.Linear(output_size, output_size, bias=False).to(device)
        # self.linear4 = nn.Linear(output_size, output_size, bias=False).to(device)
        # self.linear5 = nn.Linear(output_size, output_size, bias=False).to(device)

    def forward(self, u, h0):
        y, hn = self.rnn(u, h0)
        y = self.linear(y)
        y = self.tanh(y)
        y = self.linear2(y)
        y = self.tanh(y)
        y = self.linear3(y)
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
        weight_decay=0.0,
    ):
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
            if (epoch + 1) % 100 == 0:
                train_loss_list.append(loss_train.item())
                # Test the model
                self.eval()
                with torch.no_grad():
                    hn = test_h0
                    output_test, _ = self.forward(test_set["X"], hn)
                loss_test = loss_fn(output_test, test_set["Y"])
                test_loss_list.append(loss_test.item())
                if train_msg:
                    print(
                        "Epoch: %d / %d, Train loss: %.4e, Test loss: %.4e"
                        % (epoch + 1, epochs, loss_train.item(), loss_test.item())
                    )

                if loss_save_path is not None:
                    torch.save(
                        {
                            "train_loss_list": train_loss_list,
                            "test_loss_list": test_loss_list,
                        },
                        loss_save_path,
                    )
                    print(f"Loss saved to {loss_save_path}")
                    # Save the model
            if (epoch + 1) % 1000 == 0:
                if model_save_path is not None:
                    torch.save(self.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
            if (epoch + 1) % 5000 == 0:
                pass
                # plt.plot(output_test.detach().cpu().numpy()[0, :, 34], label="train")
                # plt.plot(test_set["Y"].detach().cpu().numpy()[0, :, 34], label="test")
                # plt.legend()
                # plt.show()
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


class LRnn(nn.Module):
    """
    We find that LRNN is not as good as RNN and LSTM in terms of
    accuracy and training speed for the state estimation problem.
    """

    def __init__(
        self, input_size=2, hidden_size=10, output_size=6, bidirectional=False
    ):
        super(LRnn, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Define layers
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

        if bidirectional:
            self.linear_00 = nn.Linear(input_size, hidden_size)
            self.linear_11 = nn.Linear(hidden_size, hidden_size)
            self.linear_2 = nn.Linear(2 * hidden_size, output_size)
        else:
            self.linear_2 = nn.Linear(hidden_size, output_size)

        # Send to device in a separate method to keep the constructor clean
        self.to_device()

    def to_device(
        self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # Send all the modules to the specified device
        self.linear_0.to(device)
        self.linear_1.to(device)
        if self.bidirectional:
            self.linear_00.to(device)
            self.linear_11.to(device)
        self.linear_2.to(device)

    def lru(self, u, h0):
        # Process the entire sequence at once
        u_transformed = self.linear_0(u)
        h1_transformed = self.linear_1(h0[0, :])

        # Recurrent calculation for forward direction using cumulative sum
        output_forward = torch.cumsum(u_transformed, dim=0) + h1_transformed

        if self.bidirectional:
            # Process the entire sequence at once for the backward direction
            u_transformed_rev = self.linear_00(torch.flip(u, [0]))
            h2_transformed = self.linear_11(h0[1, :])

            # Recurrent calculation for backward direction using cumulative sum
            output_backward = torch.cumsum(u_transformed_rev, dim=0) + h2_transformed
            output_backward = torch.flip(output_backward, [0])

            # Concatenate the forward and backward directions
            output = torch.cat((output_forward, output_backward), dim=-1)
        else:
            output = output_forward

        return output

    def forward(self, u, h0):
        y = self.lru(u, h0)
        y = self.linear_2(y)
        return y

    def train_LRNN(
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
        # torch.autograd.set_detect_anomaly(True)
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
            output_train = self.forward(train_set["X"], train_h0)
            loss_train = loss_fn(output_train, train_set["Y"])
            # print(loss_train)
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
                        pred = self.forward(input_test_X, hn)
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
        return (train_loss_list,)
