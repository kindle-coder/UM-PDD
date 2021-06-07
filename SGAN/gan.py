from tensorflow.python.keras import Model
from tensorflow.keras.optimizers import Adam

learning_rate = 0.00001


def create_gan_model(generator_model, discriminator_model):
    discriminator_model.trainable = False
    gan_input = generator_model.input
    gan_output = discriminator_model(generator_model.output)
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=0.5))

    return gan_model
