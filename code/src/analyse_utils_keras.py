"""Plot keras model history."""
import matplotlib.pyplot as plt
from keras.callbacks import History


def plot_history(history: (History, dict)):
    """Plot history.
    
    Args:
        history: History or dict, model history to plot
    Returns:
        None
    """
    acc, val_acc, loss, val_loss = (history.history['acc'], history.history['val_acc'], 
                                    history.history['loss'], history.history['val_loss']
                                   ) if isinstance(history, History) else (
        history['acc'], history['val_acc'], history['loss'], history['val_loss'])
    
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    