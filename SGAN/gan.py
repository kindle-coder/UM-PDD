from tensorflow.python.keras import Model
from tensorflow.keras.optimizers import Adam


def create_gan_model(generator_model, discriminator_model):
    gan_input = generator_model.input
    gan_output = discriminator_model(generator_model.output)
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00002, beta_1=0.5))

    return gan_model
