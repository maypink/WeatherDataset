import torch

class WeatherNet(torch.nn.Module):
    def __init__(self):
        super(WeatherNet, self).__init__()

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_block1 = torch.nn.Sequential(
            torch.nn.Linear(8112, 80),
            torch.nn.Tanh()
        )

        self.fc_block2 = torch.nn.Sequential(
            torch.nn.Linear(80, 24),
            torch.nn.Tanh()
        )

        self.fc3 = torch.nn.Linear(24, 4)

    def forward(self, x):
        x = self.conv_block1(x)

        x = self.conv_block2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc_block1(x)

        x = self.fc_block2(x)

        x = self.fc3(x)

        return x
