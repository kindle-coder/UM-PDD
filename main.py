import tensorflow as tf

from gan import create_generator_model, create_discriminator_model
from configs import configure
from datasets import get_plant_diseases_dataset
from train import train
import matplotlib.pyplot as plt

configure(enable_mixed_float16=False,
          print_device_placement=False)

# strategy = tf.distribute.get_strategy()
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

train_ds, valid_ds, test_ds = get_plant_diseases_dataset()
input_shape = train_ds.element_spec[0].shape[1:]

noise_dim = 100
epochs = 10

generator = create_generator_model(noise_dim)
discriminator = create_discriminator_model(input_shape=input_shape)


train(generator=generator,
      discriminator=discriminator,
      train_ds=train_ds,
      valid_ds=valid_ds,
      noise_dim=noise_dim,
      epochs=epochs)
