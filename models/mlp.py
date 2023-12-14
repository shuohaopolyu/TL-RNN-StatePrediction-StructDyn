import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mlp(torch.nn.Module):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add a fully connected layer
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add a Tanh activation function (except for the output layer)
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.Tanh())

        self.linear_tanh_stack = torch.nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.linear_tanh_stack(x)
        return x

    def train_MLP(self, train_set, test_set, epochs, learning_rate, model_save_path, loss_save_path, train_msg=True):
        # Defining the loss function
        loss_func = torch.nn.MSELoss(reduction='mean')
        # Setting the optimizer as Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # List for storing the loss after each epoch
        train_loss_list = []
        test_loss_list = []
        # Start training
        for epoch in range(epochs):
            self.train()
            output = self.forward(train_set["X"])
            # Calculating the training loss
            loss_train = loss_func(output, train_set["Y"])
            optimizer.zero_grad()
            loss_train.backward(retain_graph=True)
            optimizer.step()
            train_loss_list.append(loss_train.item())
            if (epoch+1) % 2000 == 0:
                train_loss_list.append(loss_train.item())
                # Calculating the test loss
                self.eval()
                with torch.no_grad():
                    test_output = self.forward(test_set["X"])
                loss_test = loss_func(test_output, test_set["Y"])
                test_loss_list.append(loss_test.item())
                # Print the loss
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
                        {
                            "train_loss_list": train_loss_list,
                            "test_loss_list": test_loss_list,
                        },
                        loss_save_path,
                    )
                    print(f"Loss saved to {loss_save_path}")                
            
        return train_loss_list, test_loss_list
