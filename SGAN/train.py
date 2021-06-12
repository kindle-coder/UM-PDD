import os
import shutil
import json

import tensorflow as tf

from tensorflow import ones
from tqdm import tqdm

import configs
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
                   unsupervised_batches_per_iteration=1,
                   save_interval=17,
                   i=0, j=0, batch_per_epoch=0
                   ):
    tensorboard_path = result_path + 'tensorboard/'
    checkpoints_path = result_path + 'checkpoints/'
    generator_samples_path = result_path + 'samples/'
    state_path = result_path + 'state.json'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        if os.path.exists(state_path):
            with open(state_path, "r") as json_file:
                state = json.loads(json_file.read())
                json_file.close()
                batch_size = state["batch_size"]
                latent_dim = state["latent_dim"]
                epochs = state["epochs"]
                batch_per_epoch = state["batch_per_epoch"]
                save_interval = state["save_interval"]
                supervised_batches_per_iteration = state["super_batches"]
                unsupervised_batches_per_iteration = state["unsuper_batches"]
                i = state["i"]
                j = state["j"]
                step = state["step"]
            last_checkpoint_dir = checkpoints_path + '{:04d}/'.format(step)
            load_models_weights(last_checkpoint_dir=last_checkpoint_dir,
                                generator=generator,
                                discriminator=discriminator,
                                classifier=classifier,
                                gan=gan)
            print("Resuming previous training")

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if not os.path.exists(generator_samples_path):
        os.makedirs(generator_samples_path)

    if batch_per_epoch == 0:
        batch_per_epoch = int(unsupervised_ds.cardinality().numpy())

    steps = int(batch_per_epoch / unsupervised_batches_per_iteration)

    train_summary_writer = tf.summary.create_file_writer(tensorboard_path + "train/")
    test_summary_writer = tf.summary.create_file_writer(tensorboard_path + "test/")

    classifier_loss = 0.0
    classifier_acc = 0.0
    discriminator_real_loss = 0.0
    discriminator_fake_loss = 0.0

    previous_steps = i * steps + j
    supervised_ds = supervised_ds.skip(previous_steps * supervised_batches_per_iteration).prefetch(configs.prefetch_no)
    unsupervised_ds = unsupervised_ds.skip(previous_steps * unsupervised_batches_per_iteration).prefetch(configs.prefetch_no)

    for i in range(epochs)[i:]:
        if profiling:
            tf.profiler.experimental.start(tensorboard_path)

        for j in tqdm(range(steps)[j+1:], desc='Epoch-{:02d}'.format(i + 1), ncols=80):
            global_steps = i * steps + j

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

            if (i != 0 or j != 0) and global_steps % save_interval == 0:
                # Test the classifier performance
                test_loss, test_acc = classifier.evaluate(test_ds.take(10))
                with test_summary_writer.as_default(step=global_steps):
                    tf.summary.scalar('classifier_loss', test_loss)
                    tf.summary.scalar('classifier_acc', test_acc)

                with train_summary_writer.as_default(step=global_steps):
                    tf.summary.scalar('classifier_loss', classifier_loss)
                    tf.summary.scalar('classifier_acc', classifier_acc)
                    tf.summary.scalar('discriminator_real_loss', discriminator_real_loss)
                    tf.summary.scalar('discriminator_fake_loss', discriminator_fake_loss)
                    tf.summary.scalar('gan_loss', gan_loss)

                save_generated_images(path=generator_samples_path,
                                      generator=generator,
                                      latent_dim=latent_dim,
                                      step=global_steps,
                                      no_of_samples=9)

                save_checkpoint(generator=generator,
                                discriminator=discriminator,
                                classifier=classifier,
                                gan=gan,
                                checkpoints_path=checkpoints_path,
                                step=global_steps)

                save_state(state_path=state_path,
                           batch_size=batch_size,
                           latent_dim=latent_dim,
                           epochs=epochs,
                           batch_per_epoch=batch_per_epoch,
                           save_interval=save_interval,
                           super_batches=supervised_batches_per_iteration,
                           unsuper_batches=unsupervised_batches_per_iteration,
                           step=global_steps,
                           i=i,
                           j=j)

        if profiling:
            tf.profiler.experimental.stop()


def save_checkpoint(generator, discriminator, classifier, gan, checkpoints_path, step):
    save_dir_path = checkpoints_path + '{:04d}/'.format(step)
    generator_checkpoint_path = save_dir_path + 'generator.h5'.format(step)
    discriminator_checkpoint_path = save_dir_path + 'discriminator.h5'.format(step)
    classifier_checkpoint_path = save_dir_path + 'classifier.h5'.format(step)
    gan_checkpoint_path = save_dir_path + 'gan.h5'.format(step)

    generator.save(generator_checkpoint_path)
    discriminator.save(discriminator_checkpoint_path)
    classifier.save(classifier_checkpoint_path)
    gan.save(gan_checkpoint_path)


def save_state(state_path,
               batch_size, latent_dim, epochs,
               batch_per_epoch, save_interval, super_batches,
               unsuper_batches, step, i, j):
    state = {
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_per_epoch": batch_per_epoch,
        "save_interval": save_interval,
        "super_batches": super_batches,
        "unsuper_batches": unsuper_batches,
        "step": step,
        "i": i,
        "j": j
    }

    with open(state_path, "w") as json_file:
        json_file.write(json.dumps(state))


def load_models_weights(last_checkpoint_dir, generator, classifier, discriminator, gan):
    generator_file_path = last_checkpoint_dir + "generator.h5"
    classifier_file_path = last_checkpoint_dir + "classifier.h5"
    discriminator_file_path = last_checkpoint_dir + "discriminator.h5"
    gan_file_path = last_checkpoint_dir + "gan.h5"

    temp_model = tf.keras.models.load_model(generator_file_path)
    generator.set_weights(temp_model.get_weights())

    temp_model = tf.keras.models.load_model(classifier_file_path)
    classifier.set_weights(temp_model.get_weights())

    temp_model = tf.keras.models.load_model(discriminator_file_path)
    discriminator.set_weights(temp_model.get_weights())

    temp_model = tf.keras.models.load_model(gan_file_path)
    gan.set_weights(temp_model.get_weights())

