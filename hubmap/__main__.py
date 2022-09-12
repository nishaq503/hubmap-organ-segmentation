import json
import logging
import os
import pathlib

import keras.callbacks
import segmentation_models
from segmentation_models import losses
from segmentation_models import metrics

from hubmap.data import datagen
from hubmap.data import preparse
from hubmap.utils import constants
from hubmap.utils import helpers
from hubmap.utils import paths

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
    logger.info(f'Starting to tile the training images ...')
    preparse.tile_all_images(paths.TRAIN_CSV)

backbone = 'vgg16'
assert backbone in constants.BACKBONES, f'backbone must be one of {constants.BACKBONES}'
saved_models = paths.SAVED_MODELS.joinpath(backbone)
saved_models.mkdir(exist_ok=True)

logger.info('Initializing model ...')
model: keras.Model = segmentation_models.Unet(
    backbone_name=backbone,
    input_shape=(constants.TILE_SIZE, constants.TILE_SIZE, 3),
    # classes=1,
    encoder_weights='imagenet',
    # encoder_freeze=True,
)

model.compile(
    optimizer='adam',
    loss=losses.dice_loss,
    metrics=[metrics.iou_score, metrics.f1_score],
)

logger.info('Creating Data Generator ...')
data = datagen.HubMapData(paths.TRAIN_CSV, batch_size=32)

# model.summary(line_length=160)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=saved_models.joinpath('model-{epoch:04d}-{loss:.4f}.h5'),
        monitor='loss',
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-4,
        patience=4,
        verbose=1,
        restore_best_weights=True,
    ),
]

saved_paths = list(sorted(path for path in saved_models.iterdir() if path.name.startswith('model-')))
if len(saved_paths) > 0:
    logger.info(f'loading model from before ...')
    last_path = saved_paths[-1]
    model.load_weights(last_path)
    initial_epoch = int(last_path.name.split('-')[1])
else:
    initial_epoch = 0

history = model.fit(
    x=data,
    epochs=128,
    callbacks=callbacks,
    initial_epoch=initial_epoch,
)

logger.info(f'saving final model ...')
model.save(
    filepath=saved_models.joinpath('final_model'),
)
with open(saved_models.joinpath('history.json'), 'w') as writer:
    json.dump(history.history, writer, indent=4)

model.predict()
