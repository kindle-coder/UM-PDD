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
from configs import configure
from Utils.datasets import get_plant_diseases_dataset, normalize_image
from Utils.enums import User, Environment, Accelerator

# configuration
user = User.Arash
environment = Environment.GoogleColab
accelerator = Accelerator.GPU

batch_size = 64
latent_dim = 100
epochs = 10
supervised_samples_ratio = 0.05
save_interval = 17
super_batches = 1
unsuper_batches = 1
prefetch_no = tf.data.AUTOTUNE
eager_execution = True

# Parsing Arguments
for arg in sys.argv:
    if arg.lower().__contains__("user"):
        param = arg[arg.index("=") + 1:]
        if param.lower() == "arash":
            user = User.Arash
        elif param.lower() == "kinza":
            user = User.Kinza
    if arg.lower().__contains__("envi"):
        param = arg[arg.index("=") + 1:]
        if param.lower() == "local":
            environment = Environment.Local
        elif param.lower() == "colab":
            environment = Environment.GoogleColab
        elif param.lower() == "research":
            environment = Environment.GoogleResearch
    if arg.lower().__contains__("accel"):
        param = arg[arg.index("=") + 1:]
        if param.lower() == "gpu":
            accelerator = Accelerator.GPU
        elif param.lower() == "tpu":
            accelerator = Accelerator.TPU
    if arg.lower().__contains__("batch"):
        param = arg[arg.index("=") + 1:]
        batch_size = int(param)
    if arg.lower().__contains__("epoch"):
        param = arg[arg.index("=") + 1:]
        epochs = int(param)
    if arg.lower().__contains__("sample_ratio"):
        param = arg[arg.index("=") + 1:]
        supervised_samples_ratio = float(param)
    if arg.lower().__contains__("save_interval"):
        param = arg[arg.index("=") + 1:]
        save_interval = int(param)
    if arg.lower().__contains__("super_batches"):
        param = arg[arg.index("=") + 1:]
        super_batches = int(param)
    if arg.lower().__contains__("unsuper_batches"):
        param = arg[arg.index("=") + 1:]
        unsuper_batches = int(param)
    if arg.lower().__contains__("eager"):
        param = arg[arg.index("=") + 1:]
        eager_execution = bool(param)

print(user)
print(environment)
print(accelerator)
print("Batch Size: ", batch_size)
print("Epochs: ", epochs)
print("Supervised Ratio: ", supervised_samples_ratio)
print("Save Interval: ", save_interval)
print("Supervised Batches per Interval: ", super_batches)
print("Unsupervised Batches per Interval: ", unsuper_batches)
print("Eager Execution: ", eager_execution)

# Configuring TensorFlow
configure(enable_mixed_float16=False,
          print_device_placement=False,
          enable_eager_execution=eager_execution)

strategy = tf.distribute.get_strategy()

if environment == Environment.Local:
    accelerator = Accelerator.GPU

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
    train.result_path = '/content/drive/MyDrive/Share/UM-PDD/results/'

if accelerator == Accelerator.TPU:
    batch_size = 256

unsupervised_ds, supervised_ds, test_ds = get_plant_diseases_dataset(batch_size, supervised_samples_ratio)

input_shape = unsupervised_ds.element_spec[0].shape[1:]
no_of_classes = len(unsupervised_ds.class_names)

unsupervised_ds = unsupervised_ds.map(normalize_image,
                                      num_parallel_calls=multiprocessing.cpu_count()).prefetch(prefetch_no)
supervised_ds = supervised_ds.map(normalize_image,
                                  num_parallel_calls=multiprocessing.cpu_count()).prefetch(prefetch_no)
test_ds = test_ds.map(normalize_image,
                      num_parallel_calls=multiprocessing.cpu_count()).prefetch(prefetch_no)

with strategy.scope():
    generator_model = create_generator_model(latent_dim)
    discriminator_model, classifier_model = create_discriminator_models(input_shape, no_of_classes)
    gan_model = create_gan_model(generator_model, discriminator_model)

    classifier_model.summary()

    start_training(generator=generator_model,
                   discriminator=discriminator_model,
                   classifier=classifier_model,
                   gan=gan_model,
                   batch_size=batch_size,
                   unsupervised_ds=unsupervised_ds,
                   supervised_ds=supervised_ds,
                   test_ds=test_ds,
                   epochs=epochs,
                   latent_dim=latent_dim,
                   supervised_batches_per_iteration=super_batches,
                   unsupervised_batches_per_iteration=unsuper_batches,
                   save_interval=save_interval)
