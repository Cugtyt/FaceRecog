"""
Resnet keras code.

Use resnet to implement face recognition algo.
"""
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D
from keras.optimizers import Optimizer


def identity_block(input_tensor, filters: int):
    """The identity block is the block that has no conv layer at shortcut.
    
    Args:
        input_tensor: input
        filters: int, filters
    Returns:
        x: output
    """
    x = Conv2D(filters, 3, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
#     x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, filters):
    """A block that has a conv layer at shortcut.
    
    Args:
        input_tensor: input
        filters: list, filters
    Returns:
        x: output
    """
    filter1, filter2 = filters

    x = Conv2D(filter1, 3, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, 3)(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(filter2, 3)(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet(input_shape: tuple=(128, 128, 3), classes: int=10, optimizer: (str, Optimizer)='adam'):
    """Resnet model.
    
    Args:
        input_shape: tuple, image shape, channel last, default (128, 128, 3)
        classes: int, image classes, default 10
        optimizer: str or Optimizer, optimizer to compile model, default adam
    Returns:
        model: Model, resnet model
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = identity_block(x, 32)
    x = MaxPooling2D()(x)
#     x = Dropout(0.5)(x)
    x = conv_block(x, [32, 64])
    x = MaxPooling2D()(x)
#     x = Dropout(0.5)(x)
    x = conv_block(x, [64, 128])
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
