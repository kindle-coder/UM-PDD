import tensorflow as tf

from SGAN.discriminator import create_discriminator_models
from SGAN.gan import create_gan_model
from SGAN.generator import create_generator_model
from SGAN.train import train
from configs import configure
from Utils.datasets import get_plant_diseases_dataset

batch_size = 32
latent_dim = 100
epochs = 10
supervised_samples_ratio = 0.01

strategy = tf.distribute.get_strategy()
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

unsupervised_ds, supervised_ds, test_ds = get_plant_diseases_dataset(batch_size, supervised_samples_ratio)

input_shape = unsupervised_ds.element_spec[0].shape[1:]
no_of_classes = len(unsupervised_ds.class_names)

with strategy.scope():
    generator_model = create_generator_model(latent_dim)
    discriminator_model, classifier_model = create_discriminator_models(input_shape, no_of_classes)
    gan_model = create_gan_model(generator_model, discriminator_model)

    train(generator=generator_model,
          discriminator=discriminator_model,
          classifier=classifier_model,
          gan=gan_model,
          batch_size=batch_size,
          unsupervised_ds=unsupervised_ds.prefetch(tf.data.AUTOTUNE),
          supervised_ds=supervised_ds.prefetch(tf.data.AUTOTUNE),
          epochs=epochs,
          latent_dim=latent_dim)
