import gc
import pathlib
import typing

import filepattern
import keras.utils
import numpy
import pandas

from hubmap.utils import constants
from hubmap.utils import paths


class HubMapData(keras.utils.Sequence):
    def __init__(
            self,
            csv_path: pathlib.Path,
            batch_size: int = 128,
            shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        df = pandas.read_csv(csv_path)
        self.ids = df.id.to_list()

        del df
        gc.collect()

        self.train_tiles_dir = paths.TRAIN_TILES
        self.tile_pattern = '{p+}_x{xx}_y{yy}_c{c}.npy'
        tile_fp = filepattern.FilePattern(self.train_tiles_dir, self.tile_pattern)

        self.train_masks_dir = paths.TRAIN_MASKS
        self.mask_pattern = '{p+}_x{xx}_y{yy}.npy'
        mask_fp = filepattern.FilePattern(self.train_masks_dir, self.mask_pattern)

        # self.tile_paths = [file[0]['file'] for file in tile_fp(group_by=['p', 'x', 'y'])]
        # self.mask_tiles = [file[0]['file'] for file in mask_fp()]
        self.tile_paths: typing.List[typing.Tuple[pathlib.Path, typing.List[pathlib.Path]]] = [
            (masks[0]['file'], [tile['file'] for tile in tiles])
            for masks, tiles in zip(mask_fp(), tile_fp(group_by=['c']))
        ]

        self.num_batches = len(self.tile_paths) // self.batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            numpy.random.shuffle(self.tile_paths)
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
        masks = numpy.empty((self.batch_size, constants.TILE_SIZE, constants.TILE_SIZE, 1), dtype=numpy.uint8)

        for i, index in enumerate(batch_indices):
            mask_path, tile_paths = self.tile_paths[index]
            masks[i, :, :, 0] = numpy.load(str(mask_path))

            for c, path in enumerate(tile_paths):
                tiles[i, :, :, c] = numpy.load(str(path))

        return tiles, masks


__all__ = [
    'HubMapData',
]
