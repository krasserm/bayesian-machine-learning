import numpy as np

import keras

from keras import backend as K
from keras import layers
from keras.models import Model, Sequential

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Shape of MNIST images
image_shape = (28, 28, 1)

def create_encoder(latent_dim):
    '''
    Creates a convolutional encoder model for MNIST images.
    
    - Input for the created model are MNIST images. 
    - Output of the created model are the sufficient statistics
      of the variational distriution q(t|x;phi), mean and log 
      variance. 
    '''
    encoder_iput = layers.Input(shape=image_shape, name='image')
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_iput)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    t_mean = layers.Dense(latent_dim, name='t_mean')(x)
    t_log_var = layers.Dense(latent_dim, name='t_log_var')(x)

    return Model(encoder_iput, [t_mean, t_log_var], name='encoder')

def create_decoder(latent_dim):
    '''
    Creates a (de-)convolutional decoder model for MNIST images.
    
    - Input for the created model are latent vectors t.
    - Output of the model are images of shape (28, 28, 1) where
      the value of each pixel is the probability of being white.
    '''
    decoder_input = layers.Input(shape=(latent_dim,), name='t')
    
    x = layers.Dense(12544, activation='relu')(decoder_input)
    x = layers.Reshape((14, 14, 64))(x)
    x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='image')(x)
    
    return Model(decoder_input, x, name='decoder')

def sample(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.
    
    Args:
        args: sufficient statistics of the variational distribution.
        
    Returns:
        Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon

def create_sampler():
    '''
    Creates a sampling layer.
    '''
    return layers.Lambda(sample, name='sampler')

def create_predictor_linear(latent_dim):
    '''
    Creates a regressor that estimates digit values 
    from latent variables.
    '''
    predictor_input = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(128, activation='relu')(predictor_input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(1, activation='linear')(x)

    return Model(predictor_input, x, name='predictor')

def create_classifier():
    '''
    Creates a classifier that predicts digit labels 
    from digit images.
    '''
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def plot_nll(gx, gy, nll):
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(hspace=0.4)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        gz = nll(i).reshape(gx.shape)
        im = plt.contourf(gx, gy, gz, 
                          cmap='coolwarm', 
                          norm=LogNorm(), 
                          levels=np.logspace(0.2, 1.8, 100))
        plt.title(f'Target = {i}')
    
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, fig.add_axes([0.82, 0.13, 0.02, 0.74]), 
                 ticks=np.logspace(0.2, 1.8, 11), format='%.2f', 
                 label='Negative log likelihood')