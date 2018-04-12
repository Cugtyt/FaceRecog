"""Plot pytorch model history."""
import matplotlib.pyplot as plt


def plot_history(history: dict):
    """Plot history.
    
    Args:
        history: dict, model history to plot
    Returns:
        None
    """
    train_acc = history['acc']['train']
    val_acc = history['acc']['val']
    train_loss = history['loss']['train']
    val_loss = history['loss']['val']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'b', label='training acc')
    plt.plot(epochs, val_acc, 'y', label='val acc')
    plt.title('training and val acc')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, 'b', label='training loss')
    plt.plot(epochs, val_loss, 'y', label='val loss')
    plt.title('training and val loss')
    plt.legend()

    plt.show()