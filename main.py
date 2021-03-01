import torch
import torchvision
import data_preparation
from WeatherDataset import WeatherDataset
from WeatherNet import WeatherNet
from training import training
from vizualization import vizualization_plt

data_preparation.reprod_init()

x_train, x_test, y_train, y_test = data_preparation.get_data()

composed = torchvision.transforms.Compose([data_preparation.resize, data_preparation.to_tensor])

train = WeatherDataset(lists=(x_train, y_train), transform=composed)
test = WeatherDataset(lists=(x_test, y_test), transform=composed)

train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=224, shuffle=True)

weather_net = WeatherNet()

test_loss_history, test_accuracy_history = training(weather_net, train_loader, test_loader)

vizualization_plt(test_loss_history, test_accuracy_history)