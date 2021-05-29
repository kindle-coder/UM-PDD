import tensorflow as tf
import os

from tensorflow.keras import mixed_precision


def configure(enable_xla: bool = True,
              print_device_placement: bool = False,
              enable_eager_execution: bool = True,
              only_cpu: bool = False,
              enable_memory_growth: bool = True,
              enable_mixed_float16: bool = False):
    # Configurations
    #########################################################
    # To enable xla compiler
    if enable_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    #########################################################
    # To print out on which device operation is taking place
    if print_device_placement:
        tf.debugging.set_log_device_placement(True)
    #########################################################
    # To disable eager execution and use graph functions
    if not enable_eager_execution:
        tf.compat.v1.disable_eager_execution()
    #########################################################
    # To disable GPUs
    if only_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #########################################################
    # Setting memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if enable_memory_growth and gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except "Invalid Device":
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    #########################################################
    # Create 2 virtual GPUs with 1GB memory each
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
    #              tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    #########################################################
    # Using mixed_precision to activate Tensor Cores
    if enable_mixed_float16:
        mixed_precision.set_global_policy('mixed_float16')
    #########################################################
    # Configurations
