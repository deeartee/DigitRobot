from torch import nn


class DigitConv(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_a = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dense_a = nn.Linear(32 * 7 * 7, 128)
        self.dense_b = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()
        self.regularize = nn.Dropout(0.3)

    def forward(self, inputs):
        x = self.pool(self.activation(self.conv_a(inputs)))
        x = self.pool(self.activation(self.conv_b(x)))
        x = x.flatten(1)
        x = self.regularize(self.activation(self.dense_a(x)))
        return self.dense_b(x)