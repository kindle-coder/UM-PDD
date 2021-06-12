import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import multiprocessing

import tensorflow as tf

from SGAN.discriminator import create_discriminator_models
from SGAN.gan import create_gan_model
from SGAN.generator import create_generator_model
from SGAN.train import start_training
from Utils import datasets
from SGAN import train
import configs
from configs import configure, parse_args
from Utils.datasets import get_plant_diseases_dataset, normalize_image
from Utils.enums import User, Environment, Accelerator


parse_args()

print(configs.user)
print(configs.environment)
print(configs.accelerator)
print("Batch Size: ", configs.batch_size)
print("Epochs: ", configs.epochs)
print("Supervised Ratio: ", configs.supervised_samples_ratio)
print("Save Interval: ", configs.save_interval)
print("Supervised Batches per Interval: ", configs.super_batches)
print("Unsupervised Batches per Interval: ", configs.unsuper_batches)
print("Eager Execution: ", configs.eager_execution)
print("Print Model Summery: ", configs.model_summery)


# Configuring TensorFlow
configure(enable_mixed_float16=False,
          print_device_placement=False,
          enable_eager_execution=configs.eager_execution)

strategy = tf.distribute.get_strategy()

if configs.environment == Environment.Local:
    configs.accelerator = Accelerator.GPU

if configs.accelerator == Accelerator.TPU and \
        (configs.environment == Environment.GoogleColab or configs.environment == Environment.GoogleResearch):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("TPUs: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)

if configs.environment == Environment.GoogleColab and configs.accelerator == Accelerator.GPU:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    datasets.dataset_path = '/content/drive/MyDrive/Share/UM-PDD/dataset/'
    train.result_path = '/content/drive/MyDrive/Share/UM-PDD/results/'

if configs.environment == Environment.GoogleColab and configs.accelerator == Accelerator.TPU:
    datasets.dataset_path = '/content/dataset/'
    train.result_path = '/content/results/'


unsupervised_ds, supervised_ds, test_ds = get_plant_diseases_dataset(configs.batch_size, configs.supervised_samples_ratio)

input_shape = unsupervised_ds.element_spec[0].shape[1:]
no_of_classes = len(unsupervised_ds.class_names)

unsupervised_ds = unsupervised_ds.map(normalize_image,
                                      num_parallel_calls=multiprocessing.cpu_count()).prefetch(configs.prefetch_no)
supervised_ds = supervised_ds.map(normalize_image,
                                  num_parallel_calls=multiprocessing.cpu_count()).prefetch(configs.prefetch_no)
test_ds = test_ds.map(normalize_image,
                      num_parallel_calls=multiprocessing.cpu_count()).prefetch(configs.prefetch_no)

with strategy.scope():
    generator_model = create_generator_model(configs.latent_dim)
    discriminator_model, classifier_model = create_discriminator_models(input_shape, no_of_classes)
    gan_model = create_gan_model(generator_model, discriminator_model)

    if configs.model_summery:
        print("Classifier Summery:")
        classifier_model.summary()
        print("Discriminator Summery:")
        discriminator_model.summary()
        print("Generator Summery:")
        generator_model.summary()
        print("GAN Summery:")
        gan_model.summary()

    start_training(generator=generator_model,
                   discriminator=discriminator_model,
                   classifier=classifier_model,
                   gan=gan_model,
                   batch_size=configs.batch_size,
                   unsupervised_ds=unsupervised_ds,
                   supervised_ds=supervised_ds,
                   test_ds=test_ds,
                   epochs=configs.epochs,
                   latent_dim=configs.latent_dim,
                   supervised_batches_per_iteration=configs.super_batches,
                   unsupervised_batches_per_iteration=configs.unsuper_batches,
                   save_interval=configs.save_interval)
