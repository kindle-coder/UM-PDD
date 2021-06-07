import matplotlib
from matplotlib import pyplot
from tensorflow import zeros
from numpy.random import randn
import numpy as np


def generate_latent_points(latent_dim, no_of_samples):
    # generate points in the latent space
    latent = randn(latent_dim * no_of_samples)
    # reshape into a batch of inputs for the network
    latent = latent.reshape(no_of_samples, latent_dim)
    return latent


def generate_fake_samples(generator, latent_dim, no_of_samples):
    # generate points in latent space
    latent = generate_latent_points(latent_dim, no_of_samples)
    # predict outputs
    images = generator.predict(latent)
    # create class labels
    labels = zeros(no_of_samples)
    return images, labels


def save_generated_images(path, generator, latent_dim, no_of_samples, step):
    images, _ = generate_fake_samples(generator, latent_dim, no_of_samples)
    images = images.astype(np.float32)
    images = (images + 1) / 2.0
    images = images / images.max()
    images = (images * 255).astype('int')

    fig = pyplot.gcf()
    fig.set_size_inches(18.5*5, 10.5*5)

    for i in range(25):
        # define subplot
        pyplot.subplot(5, 5, i + 1)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i, :, :, :])
    # save plot to file
    filename = path + 'generated_%04d.png' % (step + 1)
    pyplot.savefig(filename)
    pyplot.close()
