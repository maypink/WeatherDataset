import torch
from torch.utils.tensorboard import SummaryWriter

def training(weather_net, train_loader, test_loader):

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(weather_net.parameters(), lr=1.0e-3)

    batches_per_epoch = len(train_loader)

    writer = SummaryWriter('runs/exp4')

    test_accuracy_history = []
    test_loss_history = []

    train_accuracy_history = []
    train_loss_history = []

    print('Training had started...')

    for epoch in range(500):
        for batch_idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()

            X_batch = data.permute(0, 3, 1, 2)
            y_batch = label

            preds = weather_net.forward(X_batch.float())

            loss_value = loss(preds, y_batch)
            writer.add_scalar('Loss/train', loss_value, batch_idx + batches_per_epoch * epoch)
            train_loss_history.append(loss_value)
            train_accuracy = (preds.argmax(dim=1) == y_batch).float().mean()
            train_accuracy_history.append(train_accuracy)
            writer.add_scalar('Accuracy/train', train_accuracy, batch_idx + batches_per_epoch * epoch)
            loss_value.backward()

            optimizer.step()

        for batch_idx, (data, label) in enumerate(test_loader):
            test_preds = weather_net.forward(data.permute(0, 3, 1, 2).float())
            with torch.no_grad():
                cur_loss = loss(test_preds, label)
                test_loss_history.append(cur_loss)
                writer.add_scalar('Loss/test', cur_loss, batches_per_epoch * epoch)

            accuracy = (test_preds.argmax(dim=1) == label).float().mean()
            writer.add_scalar('Accuracy/test', accuracy, batches_per_epoch * epoch)
            test_accuracy_history.append(accuracy)

            if epoch % 10 == 0 and epoch != 0:
                print('Average value of the loss function over the last 10 epochs on test data is ',
                      sum(test_loss_history[-10:]) / 10)
                print('Average value of the accuracy over the last 10 epochs on test data is ',
                      sum(test_accuracy_history[-10:]) / 10,
                      end='\n')
                print('Average value of the loss function over the last 10 epochs on train data is ', sum(train_loss_history[-10:]) / 10)
                print('Average value of the accuracy over the last 10 epochs on train data is ', sum(train_accuracy_history[-10:]) / 10,
                      end='\n\n')
    return test_loss_history, test_accuracy_history, train_loss_history, train_accuracy_history
