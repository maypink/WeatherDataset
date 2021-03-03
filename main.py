import torch
import torchvision
import data_preparation
from weather_dataset import WeatherDataset
from weather_net import WeatherNet
from training import training
from vizualization import vizualization_plt

data_preparation.reprod_init()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = data_preparation.get_data()

composed = torchvision.transforms.Compose([data_preparation.resize, data_preparation.to_tensor])

train = WeatherDataset(lists=(x_train, y_train), transform=composed)
test = WeatherDataset(lists=(x_test, y_test), transform=composed)

train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=224, shuffle=True)

weather_net = WeatherNet()

weather_net = weather_net.to(device)

test_loss_history, test_accuracy_history, train_loss_history, train_accuracy_history = training(weather_net, train_loader, test_loader, device)

vizualization_plt(test_loss_history, test_accuracy_history, train_loss_history, train_accuracy_history)
