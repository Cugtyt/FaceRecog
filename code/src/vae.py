import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.optimizers import Optimizer


latent_dim = 10
    
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded, z_mean, z_log_var):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


def vae(img_shape: tuple=(128, 128, 3), optimizer: (str, Optimizer)='adam'):
    input_img = keras.Input(shape=img_shape)

    x = layers.Conv2D(32, 3,
                      padding='same', activation='relu')(input_img)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu',
                      strides=(2, 2))(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    
    decoder_input = layers.Input(K.int_shape(z)[1:])

    # Upsample to the correct number of units
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)

    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # We then apply then reverse operation to the initial
    # stack of convolution layers: a `Conv2DTranspose` layers
    # with corresponding parameters.
    x = layers.Conv2DTranspose(32, 3,
                               padding='same', activation='relu',
                               strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, 
                      padding='same', activation='sigmoid')(x)
    # We end up with a feature map of the same size as the original input.

    # This is our decoder model.
    decoder = Model(decoder_input, x)
    # We then apply it to `z` to recover the decoded `z`.
    z_decoded = decoder(z)
    
    # We call our custom layer on the input and the decoded output,
    # to obtain the final model output.
    y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
    
    vae = Model(input_img, y)
    vae.compile(optimizer=optimizer, loss=None)
    
    return vae