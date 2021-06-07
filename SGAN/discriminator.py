import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.python.keras import backend
from tensorflow.keras.layers import BatchNormalization

learning_rate = 0.00002


# Source: https://arxiv.org/abs/1606.03498
def unsupervised_activation(output):
    log_exp_sum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = log_exp_sum / (log_exp_sum + 1.0)
    return result


def create_discriminator_models(input_shape, no_of_classes):
    input_layer = Input(shape=input_shape)

    middle_layer = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(input_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = BatchNormalization()(middle_layer)

    middle_layer = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = BatchNormalization()(middle_layer)

    middle_layer = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = BatchNormalization()(middle_layer)

    middle_layer = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(middle_layer)
    middle_layer = LeakyReLU(alpha=0.2)(middle_layer)

    middle_layer = BatchNormalization()(middle_layer)

    middle_layer = Flatten()(middle_layer)
    middle_layer = Dropout(0.4)(middle_layer)
    middle_layer = Dense(no_of_classes)(middle_layer)

    # Supervised Model
    classification_output_layer = Activation('softmax')(middle_layer)
    classifier_model = Model(input_layer, classification_output_layer)
    classifier_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=Adam(learning_rate=learning_rate, beta_1=0.5), metrics=['accuracy'])

    # Unsupervised Mode
    discriminator_output_layer = Lambda(unsupervised_activation)(middle_layer)
    # define and compile unsupervised discriminator model
    discriminator_model = Model(input_layer, discriminator_output_layer)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, beta_1=0.5))

    return discriminator_model, classifier_model
