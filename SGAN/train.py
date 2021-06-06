import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import zeros, ones

from Utils.fake_samples import generate_fake_samples, generate_latent_points


def train(generator,
          discriminator,
          classifier,
          gan,
          unsupervised_ds,
          supervised_ds,
          epochs,
          latent_dim,
          batch_size,
          supervised_batches_per_iteration=1,
          unsupervised_batches_per_iteration=5,
          ):

    batch_per_epoch = unsupervised_ds.cardinality()

    steps = int(batch_per_epoch * epochs / unsupervised_batches_per_iteration)

    for i in range(steps):
        # update supervised discriminator
        supervised_subset = supervised_ds.take(supervised_batches_per_iteration)
        for image_batch in supervised_subset:
            classifier_loss, classifier_acc = classifier.train_on_batch(image_batch[0], image_batch[1])

        # update unsupervised discriminator
        discriminator.trainable = True
        unsupervised_subset = unsupervised_ds.take(unsupervised_batches_per_iteration)
        for image_batch in unsupervised_subset:
            no_of_samples = len(image_batch[0])
            labels = ones(no_of_samples)
            discriminator_real_loss = discriminator.train_on_batch(image_batch[0], labels)

        for _ in range(unsupervised_batches_per_iteration):
            images_fake, labels_fake = generate_fake_samples(generator, latent_dim, batch_size)
            discriminator_fake_loss = discriminator.train_on_batch(images_fake, labels_fake)

        # update generator
        discriminator.trainable = False
        gan_input = generate_latent_points(latent_dim, batch_size)
        gan_labels = ones(batch_size)
        gan_loss = gan.train_on_batch(gan_input, gan_labels)
