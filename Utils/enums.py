from enum import Enum


class User(Enum):
    Arash = 1
    Kinza = 2


class Environment(Enum):
    Local = 1
    GoogleColab = 2
    GoogleResearch = 3


class Engine(Enum):
    TensorFlow = 1
    PlaidML = 2


class Accelerator(Enum):
    GPU = 1
    TPU = 2
