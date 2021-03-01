import matplotlib.pyplot as plt

def vizualization_plt(test_loss_history, test_accuracy_history, train_loss_history, train_accuracy_history):
    plt.plot(test_loss_history)
    plt.plot(test_accuracy_history)
    plt.show()

    plt.plot(train_loss_history)
    plt.plot(train_accuracy_history)
    plt.show()
