import logging
import pathlib

import keras
import segmentation_models
from segmentation_models import losses
from segmentation_models import metrics

from hubmap.data import datagen
from hubmap.data import preparse
from hubmap.utils import constants
from hubmap.utils import helpers
from hubmap.utils import paths

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger('hubmap.main')


logger.info('Hello from hubmap!')

for path in paths.INITIAL_PATHS:
    if isinstance(path, pathlib.PosixPath):
        if path.exists():
            logger.debug(f'Found path: {path} ...')
        else:
            message = f'Path not found {path} ...'
            logger.error(message)
            raise ValueError(f'Path not found {path} ...')
else:
    logger.info(f'Found all initially required paths ...')


rerun = False  # TODO: Make this a commandline arg
if rerun:
    logger.info(f'Starting to tile the training images')
    preparse.tile_all_images(paths.TRAIN_CSV)

data = datagen.HubMapData(paths.TRAIN_CSV)

model: keras.Model = segmentation_models.Unet(
    backbone_name='resnet34',
    input_shape=(constants.TILE_SIZE, constants.TILE_SIZE, 3),
    classes=1,
    activation='sigmoid',
    encoder_weights='imagenet',
    # encoder_freeze=True,
)

model.compile(
    optimizer='adam',
    loss=losses.dice_loss,
    metrics=metrics.iou_score,
)

# model.summary(line_length=160)

model.fit_generator(
    generator=data,
    epochs=10,
    callbacks=None,
)
