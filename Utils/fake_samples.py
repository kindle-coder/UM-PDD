from tensorflow import zeros
from numpy.random import randn


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
