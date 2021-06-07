import os
import shutil

import tensorflow as tf

from tensorflow import ones
from tqdm import tqdm

from Utils.fake_samples import generate_fake_samples, generate_latent_points, save_generated_images

profiling = False

result_path = './results/'


def start_training(generator,
                   discriminator,
                   classifier,
                   gan,
                   unsupervised_ds,
                   supervised_ds,
                   test_ds,
                   epochs,
                   latent_dim,
                   batch_size,
                   supervised_batches_per_iteration=1,
                   unsupervised_batches_per_iteration=5,
                   ):
    tensorboard_path = result_path + 'tensorboard/'
    checkpoints_path = result_path + 'checkpoints/'
    generator_samples_path = result_path + 'generator_samples/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        result_path_old = "{0}-old".format(result_path[:-1])
        if os.path.exists(result_path_old):
            shutil.rmtree(result_path_old)
        os.rename(result_path, result_path_old)

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if not os.path.exists(generator_samples_path):
        os.makedirs(generator_samples_path)

    batch_per_epoch = unsupervised_ds.cardinality()

    steps = int(batch_per_epoch / unsupervised_batches_per_iteration)

    train_summary_writer = tf.summary.create_file_writer(tensorboard_path + "train/")
    test_summary_writer = tf.summary.create_file_writer(tensorboard_path + "test/")

    classifier_loss = 0.0
    classifier_acc = 0.0
    discriminator_real_loss = 0.0
    discriminator_fake_loss = 0.0

    for i in range(epochs):
        if profiling:
            tf.profiler.experimental.start(tensorboard_path)

        for j in tqdm(range(steps), desc='Epoch-{:02d}'.format(i + 1), ncols=80):
            # update supervised discriminator
            supervised_subset = supervised_ds.take(supervised_batches_per_iteration)
            for image_batch in supervised_subset:
                classifier_loss, classifier_acc = classifier.train_on_batch(image_batch[0], image_batch[1])

            # update unsupervised discriminator
            unsupervised_subset = unsupervised_ds.take(unsupervised_batches_per_iteration)
            for image_batch in unsupervised_subset:
                no_of_samples = len(image_batch[0])
                labels = ones(no_of_samples)
                discriminator_real_loss = discriminator.train_on_batch(image_batch[0], labels)

            for _ in range(unsupervised_batches_per_iteration):
                images_fake, labels_fake = generate_fake_samples(generator, latent_dim, batch_size)
                discriminator_fake_loss = discriminator.train_on_batch(images_fake, labels_fake)

            # update generator
            gan_input = generate_latent_points(latent_dim, batch_size)
            gan_labels = ones(batch_size)
            gan_loss = gan.train_on_batch(gan_input, gan_labels)

            if (i != 0 or j != 0) and (i * steps + j) % 17 == 0:
                # Test the classifier performance
                test_loss, test_acc = classifier.evaluate(test_ds.take(10))
                with test_summary_writer.as_default(step=i * steps + j):
                    tf.summary.scalar('classifier_loss', test_loss)
                    tf.summary.scalar('classifier_acc', test_acc)

                with train_summary_writer.as_default(step=i * steps + j):
                    tf.summary.scalar('classifier_loss', classifier_loss)
                    tf.summary.scalar('classifier_acc', classifier_acc)
                    tf.summary.scalar('discriminator_real_loss', discriminator_real_loss)
                    tf.summary.scalar('discriminator_fake_loss', discriminator_fake_loss)
                    tf.summary.scalar('gan_loss', gan_loss)

                save_generated_images(path=generator_samples_path,
                                      generator=generator,
                                      latent_dim=latent_dim,
                                      step=i * steps + j,
                                      no_of_samples=9)

        if profiling:
            tf.profiler.experimental.stop()

        # TODO Save Training States

        # TODO Save Checkpoints
