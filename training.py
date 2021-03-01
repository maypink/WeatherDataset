import torch
from torch.utils.tensorboard import SummaryWriter

def training(weather_net, train_loader, test_loader):

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(weather_net.parameters(), lr=1.0e-3)

    writer = SummaryWriter('runs/exp2')

    test_accuracy_history = []
    test_loss_history = []

    for epoch in range(500):
        for batch_idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()

            X_batch = data.permute(0, 3, 1, 2)
            y_batch = label

            preds = weather_net.forward(X_batch.float())

            loss_value = loss(preds, y_batch)
            train_accuracy = (preds.argmax(dim=1) == y_batch).float().mean()
            loss_value.backward()

            optimizer.step()

        for batch_idx, (data, label) in enumerate(test_loader):
            test_preds = weather_net.forward(data.permute(0, 3, 1, 2).float())
            with torch.no_grad():
                cur_loss = loss(test_preds, label)
                test_loss_history.append(cur_loss)
                writer.add_scalar('Loss/test', cur_loss, epoch)

            accuracy = (test_preds.argmax(dim=1) == label).float().mean()
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            test_accuracy_history.append(accuracy)

            if epoch % 10 == 0 and epoch != 0:
                print('Average value of the loss function over the last 10 epochs is ', sum(test_loss_history[-10:]) / 10)
                print('Average value of the accuracy over the last 10 epochs is ', sum(test_accuracy_history[-10:]) / 10,
                      end='\n\n')
    writer.close()
    return test_loss_history, test_accuracy_history