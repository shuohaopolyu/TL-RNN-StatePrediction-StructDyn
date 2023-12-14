import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AutoEncoder(torch.nn.Module):
    """
    :param input_size: number of input features
    :param hidden_size: number of hidden features
    """

    def __init__(self, layer_sizes):
        super().__init__()

        reverse_layersize = layer_sizes[::-1]
        reverse_layers = []
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add a fully connected layer
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add a Tanh activation function (except for the output layer)
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.Tanh())

        for i in range(len(reverse_layersize) - 1):
            # Add a fully connected layer
            reverse_layers.append(
                torch.nn.Linear(reverse_layersize[i], reverse_layersize[i + 1])
            )

            # Add a Tanh activation function (except for the output layer)
            if i < len(reverse_layersize) - 2:
                reverse_layers.append(torch.nn.Tanh())

        # Building an linear encoder with Linear
        # layer followed by Tanh activation function

        self.encoder = torch.nn.Sequential(*layers).to(device)

        # Building an linear decoder with Linear
        # layer followed by Tanh activation function

        self.decoder = torch.nn.Sequential(*reverse_layers).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_AE(
        self,
        train_set,
        test_set,
        epochs,
        learning_rate,
        model_save_path,
        loss_save_path,
        train_msg = True,
    ):
        # Define the loss function
        loss_fn = torch.nn.MSELoss(reduction="mean")
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Define the loss list
        train_loss_list = []
        test_loss_list = []
        # Start training
        for epoch in range(epochs):
            self.train()
            # Train the model
            output_train = self.forward(train_set)
            loss_train = loss_fn(output_train, train_set)
            optimizer.zero_grad()
            loss_train.backward(retain_graph=True)
            optimizer.step()
            # Print the loss
            if (epoch + 1) % 2000 == 0:
                train_loss_list.append(loss_train.item())
                self.eval()
                with torch.no_grad():
                    output_test = self.forward(test_set)
                    loss_test = loss_fn(output_test, test_set)
                    test_loss_list.append(loss_test.item())
                    if train_msg:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_train.item()}, Test Loss: {loss_test.item()}"
                        )
                    
                # Save the model
                if model_save_path is not None:
                    torch.save(self.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
                
                if loss_save_path is not None:
                    torch.save(
                        {"train_loss_list": train_loss_list, "test_loss_list": test_loss_list},
                        loss_save_path,
                    )
                    print(f"Loss saved to {loss_save_path}")
        return train_loss_list, test_loss_list
