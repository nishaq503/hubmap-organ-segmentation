import logging
import os

KAGGLE_LOG = getattr(logging, os.environ.get('KAGGLE_LOG', 'INFO'))

MAX_WORKERS = max(1, os.cpu_count() // 2)
EPSILON = 1e-12  # To avoid divide-by-zero errors
MAX_MEMORY = 8 * 1024 * 1024  # bytes
TILE_SIZE = 256 * 4


class Unset:
    """ This is a hack around type-hinting when a value cannot be set in the
    __init__ method for a class.
    https://peps.python.org/pep-0661/
    Usage:
    ```python
    class MyClass:
        def __init__(self, *args, **kwargs):
            ...
            self.__value: typing.Union[ValueType, Unset] = UNSET
        def value_setter(self, *args, **kwargs):
            ...
            self.__value = something
            return
        @property
        def value(self) -> ValueType:
            if self.__value is UNSET:
                raise ValueError(f'Please call `value_setter` on the object before using this property.')
            return self.__value
    ```
    """
    __unset = None

    def __new__(cls):
        if cls.__unset is None:
            cls.__unset = super().__new__(cls)
        return cls.__unset


UNSET = Unset()


BACKBONES = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'seresnet18',
    'seresnet34',
    'seresnet50',
    'seresnet101',
    'seresnet152',
    'seresnext50',
    'seresnext101',
    'senet154',
    'resnext50',
    'resnext101',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet169',
    'densenet201',
    'inceptionresnetv2',
    'inceptionv3',
    'mobilenet',
    'mobilenetv2',
    'efficientnetb0',
    'efficientnetb1',
    'efficientnetb2',
    'efficientnetb3',
    'efficientnetb4',
    'efficientnetb5',
    'efficientnetb6',
    'efficientnetb7',
]
