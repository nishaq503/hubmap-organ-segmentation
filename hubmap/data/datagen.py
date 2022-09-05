import gc
import pathlib

import keras.utils
import pandas

from hubmap.utils import paths


class HubMapData(keras.utils.Sequence):
    def __init__(
            self,
            csv_path: pathlib.Path,
            batch_size: int = 128,
    ):
        self.batch_size = batch_size

        df = pandas.read_csv(csv_path)
        self.ids = df.id.to_list()

        del df
        gc.collect()

        self.train_tiles_dir = paths.TRAIN_TILES
        self.tile_pattern = '{i}_x{x:02d}_y{y:02d}_c{c:1d}.npy'

        self.train_masks_dir = paths.TRAIN_MASKS
        self.mask_pattern = '{i}_x{x:02d}_y{y:02d}.npy'

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
