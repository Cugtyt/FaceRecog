"""
Vanilla CNN keras code.

Use simple ConvNet to implement face recognition algo.
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Optimizer


def vanilla_cnn_keras(input_shape: tuple=(128, 128, 3), classes: int=10, optimizer: (str, Optimizer)='adam'):
    """Implement vanilla ConvNet model.
    
    Args:
        input_shape: tuple, image shape, channel last, default (128, 128, 3)
        classes: int, image classes, default 10
        optimizer: str or Optimizer, optimizer to compile model, default adam
    Returns:
        model: Model, vanilla cnn model
    """
    model = Sequential([
        Conv2D(32, 3, activation='relu',
                      padding='same',
                      input_shape=input_shape),
        Conv2D(32, 3, padding='same', activation='relu'),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        
        Conv2D(64, 3, padding='same', activation='relu'),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        
        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(classes, activation='softmax')
    ])
   
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
