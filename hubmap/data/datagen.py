import random
import typing

import keras.utils
import numpy
import pandas

from hubmap.utils import constants
from hubmap.utils import helpers
from hubmap.utils import paths

logger = helpers.make_logger(__name__)


class HubMapData(keras.utils.Sequence):
    def __init__(
            self,
            ids: typing.List[int],
            batch_size: int = 128,
            shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ids = ids
        self.train_paths = [
            paths.TRAIN_RESIZED.joinpath(f'{image_id}.npy')
            for image_id in self.ids
        ]
        assert len(self.train_paths) > 0
        assert all((path.exists() for path in self.train_paths))

        # self.train_tiles_dir = paths.TRAIN_TILES
        # self.tile_pattern = '{p+}_x{xx}_y{yy}_c{c}.npy'
        # tile_fp = filepattern.FilePattern(self.train_tiles_dir, self.tile_pattern)
        #
        # self.train_masks_dir = paths.TRAIN_MASKS
        # self.mask_pattern = '{p+}_x{xx}_y{yy}.npy'
        # mask_fp = filepattern.FilePattern(self.train_masks_dir, self.mask_pattern)
        #
        # self.tile_paths: typing.List[typing.Tuple[pathlib.Path, typing.List[pathlib.Path]]] = [
        #     (mask[0]['file'], [tile['file'] for tile in tiles])
        #     for mask, tiles in zip(mask_fp(), tile_fp(group_by=['c']))
        #     if mask[0]['p'] in self.ids
        # ]

        self.num_batches = len(self.ids) // self.batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            numpy.random.shuffle(self.train_paths)
        return

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int):
        batch_start = index * self.batch_size
        batch_stop = batch_start + self.batch_size
        batch_indices = list(range(batch_start, batch_stop))
        return self.__data_generation(batch_indices)

    def __data_generation(self, batch_indices: typing.List[int]) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        tiles = numpy.empty((self.batch_size, constants.TILE_SIZE, constants.TILE_SIZE, 3), dtype=numpy.float32)
        masks = numpy.empty((self.batch_size, constants.TILE_SIZE, constants.TILE_SIZE, 1), dtype=numpy.float32)

        for i, index in enumerate(batch_indices):
            image = numpy.load(str(self.train_paths[index]))

            tiles[i, :, :, :] = image[..., :3]
            masks[i, :, :, 0] = image[..., 3]

        return tiles, masks


def train_valid_split(
        seed: int,
        valid_fraction: float,
        batch_size: int,
        shuffle: bool,
) -> typing.Tuple[HubMapData, HubMapData]:
    if not (0 < valid_fraction < 1):
        raise ValueError(f'`valid_fraction` must be a `float` in the (0, 1) range. Got {valid_fraction:.2e} instead.')

    helpers.seed_everything(seed)

    ids: typing.List[int] = pandas.read_csv(paths.TRAIN_CSV).id.to_list()
    random.shuffle(ids)

    valid_num = max(1, int(valid_fraction * len(ids)))

    logger.info(f'Creating dataloaders with {len(ids) - valid_num} training images and {valid_num} validation images ...')
    training_data = HubMapData(
        ids=ids[valid_num:],
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_data = HubMapData(
        ids=ids[:valid_num],
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return training_data, validation_data


__all__ = [
    'HubMapData',
    'train_valid_split',
]
