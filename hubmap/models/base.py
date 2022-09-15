import json
import pathlib
import typing

import keras
import keras.callbacks
import numpy
import segmentation_models
from segmentation_models import losses
from segmentation_models import metrics

from hubmap.data import datagen
from hubmap.utils import constants
from hubmap.utils import helpers

logger = helpers.make_logger(__name__)


class HubMap(keras.Model):
    def __init__(
            self,
            backbone: str,
            pretrained: bool,
    ):
        if backbone not in constants.BACKBONES:
            raise ValueError(f'`backbone` must be one of {constants.BACKBONES}. Got {backbone} instead.')

        logger.info('Initializing HubMap model ...')
        super().__init__()

        self.backbone = backbone
        self.model_name = f'unet-{self.backbone}'
        self.encoder_weights = 'imagenet' if pretrained else None
        self.callbacks = list()
        self.history = None

        self.model: keras.Model = segmentation_models.Unet(
            backbone_name=backbone,
            input_shape=(constants.TILE_SIZE, constants.TILE_SIZE, 3),
            encoder_weights=self.encoder_weights,
        )

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training=training, mask=mask)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def default_compile(self):
        self.compile(
            optimizer='adam',
            loss=losses.dice_loss,
            metrics=[metrics.iou_score, metrics.f1_score],
        )
        return

    def summary(self, *args, **kwargs):
        self.model.summary(*args, **kwargs)
        return

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def default_fit(
            self,
            training_data: datagen.HubMapData,
            epochs: int,
            validation_data: typing.Optional[datagen.HubMapData],
            saved_models_dir: pathlib.Path,
    ):
        saved_paths = list(sorted(path for path in saved_models_dir.iterdir() if path.name.startswith(self.model_name)))
        if len(saved_paths) > 0:
            last_path = saved_paths[-1]
            self.load_weights(last_path)
            initial_epoch = int(last_path.name.split('-')[2])
            logger.info(f'Loading model from epoch {initial_epoch} ...')
        else:
            logger.info('Starting training from scratch ...')
            initial_epoch = 0

        monitor = 'loss' if validation_data is None else 'val_loss'
        name_fmt = '{epoch:04d}-{' + monitor + ':.4f}.h5'
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=saved_models_dir.joinpath(f'{self.model_name}-' + name_fmt),
                monitor=monitor,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=1e-3,
                patience=8,
                verbose=1,
                restore_best_weights=True,
            ),
        ]

        logger.info(f'Fitting the model ...')
        self.history = self.model.fit(
            x=training_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch,
        )

        logger.info(f'Saving final model ...')
        self.save(saved_models_dir.joinpath(f'{self.model_name}-final.h5'))
        with open(saved_models_dir.joinpath('history.json'), 'w') as writer:
            json.dump(self.history.history, writer, indent=4)

        return self.history

    def save(self, filepath: pathlib.Path, **kwargs):
        self.model.save(filepath=filepath, **kwargs)
        return

    def load_weights(self, filepath, **kwargs):
        self.model.load_weights(filepath, **kwargs)
        return

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def default_predict(self, image: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError


__all__ = [
    'HubMap',
]
