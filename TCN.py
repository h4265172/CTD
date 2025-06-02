import torch.nn as nn
#%%
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, stride=1, dilation=1, dropout=0.2):
        """
        Initializes the Temporal Convolutional Network (TCN).

        Args:
            num_inputs (int): The number of input features (e.g., 1 for univariate time series).
            num_channels (list): A list of integers representing the number of filters for each TCN layer.
            kernel_size (int): The size of the convolutional kernels.
            dropout (float): Dropout rate.
        """
        super(TemporalConvNet, self).__init__()

        self.layers = nn.ModuleList()
        self.num_inputs = num_inputs
        in_channels = num_inputs
        
        # Create the TCN layers
        for out_channels in num_channels:
            self.layers.append(
                TemporalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, dilation=dilation, dropout=dropout)
            )
            in_channels = out_channels

        # Final fully connected layer to output predictions
        self.fc = nn.Linear(num_channels[-1], num_inputs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.fc(x[:, :, -1])  # Take only the last time step (for sequence-to-one)
        return x


#%%
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        """
        Initializes a single block of TCN, consisting of a convolution layer, batch normalization, and dropout.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels (filters).
            kernel_size (int): The size of the convolutional kernels.
            stride (int): The stride of the convolution.
            dilation (int): The dilation factor for dilated convolutions.
            dropout (float): The dropout rate.
        """
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        # Skip connection (identity map) to prevent vanishing gradients
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)

        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Add skip connection to the output
        if residual.size(2) != x.size(2):
            min_size = min(residual.size(2), x.size(2))
            residual = residual[:, :, -min_size:]
            x = x[:, :, -min_size:]
        x = x + residual
        return x