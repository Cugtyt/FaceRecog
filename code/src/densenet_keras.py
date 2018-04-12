"""
Densenet keras code.

Use dense to implement face recognition algo.
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Concatenate
from keras.optimizers import Optimizer


def dense_block(input_tensor, filters: int, steps: int):
    """Dense block.
    
    Args:
        input_tensor: input
        filters: list, filters
        steps: int, control conv layers
    Returns:
        out: output
    """
    x1 = Conv2D(filters, 3, padding='same')(input_tensor)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)
    x2 = Concatenate(axis=3)([x1, input_tensor])
    x2 = Conv2D(filters, 3, padding='same')(x2)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)
    x3 = Concatenate(axis=3)([x2, x1, input_tensor])
    if steps == 2:
        return x3
    x3 = Conv2D(filters, 3, padding='same')(x3)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)
    out = Concatenate(axis=3)([x3, x2, x1, input_tensor])
    return out


def transition_block(input_tensor, filters: int):
    x = BatchNormalization(axis=3)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters, 1)(x)
    x = MaxPooling2D(2)(x)
    return x


def densenet(input_shape: tuple=(128, 128, 3), classes: int=10, optimizer: (str, Optimizer)='adam'):
    """Densenet model.
    
    Args:
        input_shape: tuple, image shape, channel last, default (128, 128, 3)
        classes: int, image classes, default 10
        optimizer: str or Optimizer, optimizer to compile model, default adam
    Returns:
        model: Model, densenet model
    """
    inputs = Input(shape=input_shape)
    x = dense_block(inputs, 32, 3)
    x = MaxPooling2D()(x)
    x = transition_block(x, 32)
    x = dense_block(x, 64, 2)
    x = MaxPooling2D()(x)
    x = transition_block(x, 128)
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