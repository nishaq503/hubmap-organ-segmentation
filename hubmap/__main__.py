import logging
import os
import pathlib

from hubmap import models
from hubmap.data import datagen
from hubmap.data import preparse
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

model = models.HubMap(
    backbone='resnet34',
    pretrained=True,
)
model.default_compile()
# model.summary(line_length=160)

training_data, validation_data = datagen.train_valid_split(
    seed=42,
    valid_fraction=0.1,
    batch_size=16,
    shuffle=True,
)

history = model.default_fit(
    training_data=training_data,
    epochs=128,
    validation_data=validation_data,
    saved_models_dir=paths.SAVED_MODELS.joinpath(model.backbone),
)
