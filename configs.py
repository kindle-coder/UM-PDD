import sys

import tensorflow as tf
import os

from tensorflow.keras import mixed_precision

# configuration
from Utils.enums import User, Environment, Accelerator

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
model_summery = False


def parse_args():
    # Parsing Arguments
    global user
    global environment
    global accelerator
    global batch_size
    global latent_dim
    global epochs
    global supervised_samples_ratio
    global save_interval
    global super_batches
    global unsuper_batches
    global prefetch_no
    global eager_execution
    global model_summery
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
            if param.lower().__contains__("false"):
                eager_execution = False
            else:
                eager_execution = True
        if arg.lower().__contains__("model_sum"):
            param = arg[arg.index("=") + 1:]
            if param.lower().__contains__("false"):
                model_summery = False
            else:
                model_summery = True


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
