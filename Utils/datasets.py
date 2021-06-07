import tensorflow as tf

dataset_path = './dataset/'

image_size = 256


def get_plant_diseases_dataset(batch_size, supervised_samples_ratio):

    training_path = dataset_path + "train/"
    validation_path = dataset_path + "valid/"

    unsupervised_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_path,
        validation_split=supervised_samples_ratio,
        subset="training",
        seed=5584,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    supervised_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_path,
        validation_split=supervised_samples_ratio,
        subset="validation",
        seed=5584,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,
        image_size=(image_size, image_size),
        batch_size=batch_size)

    return unsupervised_ds, supervised_ds, test_ds


def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label
