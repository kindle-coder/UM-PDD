import multiprocessing

import tensorflow as tf

from SGAN.discriminator import create_discriminator_models
from SGAN.gan import create_gan_model
from SGAN.generator import create_generator_model
from SGAN.train import train
from Utils import datasets
from SGAN import train
from configs import configure
from Utils.datasets import get_plant_diseases_dataset, normalize_image

from Utils.enums import User, Environment, Accelerator

# configuration
user = User.Arash
environment = Environment.GoogleColab
accelerator = Accelerator.GPU

batch_size = 32
latent_dim = 100
epochs = 10
supervised_samples_ratio = 0.01
prefetch_no = tf.data.AUTOTUNE

configure(enable_mixed_float16=False,
          print_device_placement=False,
          enable_eager_execution=True)

strategy = tf.distribute.get_strategy()

if accelerator == Accelerator.TPU and \
        (environment == Environment.GoogleColab or environment == Environment.GoogleResearch):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("TPUs: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)

if environment == Environment.GoogleColab and accelerator == Accelerator.GPU:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

if environment == Environment.GoogleColab:
    datasets.dataset_path = '/content/drive/MyDrive/Share/UM-PDD/dataset/'
    train.result_path = '/content/drive/MyDrive/Share/UM-PDD/result/'

if accelerator == Accelerator.TPU:
    batch_size = 256


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
          unsupervised_ds=unsupervised_ds.map(normalize_image, num_parallel_calls=multiprocessing.cpu_count()).prefetch(prefetch_no),
          supervised_ds=supervised_ds.map(normalize_image, num_parallel_calls=multiprocessing.cpu_count()).prefetch(prefetch_no),
          epochs=epochs,
          latent_dim=latent_dim)
