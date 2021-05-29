import tensorflow as tf

training_path = "./dataset/train/"
validation_path = "./dataset/valid/"

image_size = 256
batch_size = 32

seed = 5584


def get_plant_diseases_dataset():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_path,
        validation_split=0.05,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_path,
        validation_split=0.05,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    return train_ds, valid_ds, test_ds
