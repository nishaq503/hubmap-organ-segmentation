import concurrent.futures
import pathlib
import typing

import numpy
import pandas
import skimage.transform
from matplotlib import pyplot

from hubmap.utils import constants
from hubmap.utils import helpers
from hubmap.utils import paths

logger = helpers.make_logger(__name__)


def resize_all_images(csv_path: pathlib.Path):

    df = pandas.read_csv(csv_path)
    ids = df.id.to_list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=constants.MAX_WORKERS) as executor:
        futures = list()

        for i in ids:
            rle = df[df.id == i]['rle'].iloc[-1]
            futures.append(executor.submit(
                resize_image,
                image_id=i,
                rle=rle,
                shape=(constants.TILE_SIZE, constants.TILE_SIZE),
            ))

        logger.info(f'Resizing images. Progress: {0:6.2f}% ...')
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            logger.info(f'Resizing images. Progress: {100 * (i + 1) / len(ids):6.2f}% ...')
            f.result()

    return


def resize_image(
        image_id: int,
        rle: str,
        shape: typing.Tuple[int, int],
):
    path = paths.TRAIN_IMAGES.joinpath(f'{image_id}.tiff')
    image = numpy.squeeze(pyplot.imread(path))
    height, width, num_channels = image.shape

    out_image = numpy.zeros(shape=(*shape, 4), dtype=numpy.float32)
    for c in range(num_channels):
        channel = image[:, :, c].astype(numpy.float32)
        min_i, max_i = numpy.min(channel), numpy.max(channel)
        channel = (channel - min_i) / (max_i - min_i + constants.EPSILON)
        channel = skimage.transform.resize(channel, output_shape=shape, preserve_range=True)
        out_image[:, :, c] = channel

    mask = helpers.rle_to_mask(rle, height, width).astype(numpy.float32)
    mask = skimage.transform.resize(mask, output_shape=shape, preserve_range=True)
    out_image[:, :, 3] = mask

    out_path = paths.TRAIN_RESIZED.joinpath(f'{image_id}.npy')
    numpy.save(
        file=str(out_path),
        arr=out_image,
        allow_pickle=True,
        fix_imports=True,
    )
    return True


def tile_all_images(csv_path: pathlib.Path):

    df = pandas.read_csv(csv_path)
    ids = df.id.to_list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=constants.MAX_WORKERS) as executor:
        futures = list()

        for i in ids:
            rle = df[df.id == i]['rle'].iloc[-1]
            futures.append(executor.submit(
                make_all_tiles,
                i,
                rle,
                constants.TILE_SIZE,
            ))

        logger.info(f'Preparsing images. Progress: {0:6.2f}% ...')
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            logger.info(f'Preparsing images. Progress: {100 * (i + 1) / len(ids):6.2f}% ...')
            f.result()

    return


def make_all_tiles(
        image_id: int,
        rle: str,
        tile_stride: int,
):
    path = paths.TRAIN_IMAGES.joinpath(f'{image_id}.tiff')
    image = numpy.squeeze(pyplot.imread(path))
    height, width, num_channels = image.shape

    for c in range(num_channels):
        make_tiles(
            image=image[:, :, c],
            tile_stride=tile_stride,
            out_dir=paths.WORKING_DIR.joinpath('train', 'intensity'),
            image_id=image_id,
            c=c,
        )

    mask = helpers.rle_to_mask(rle, height, width)
    make_tiles(
        image=mask,
        tile_stride=tile_stride,
        out_dir=paths.WORKING_DIR.joinpath('train', 'labels'),
        image_id=image_id,
        c=None,
    )

    return True


def make_tiles(
        image: numpy.ndarray,
        tile_stride: int,
        out_dir: pathlib.Path,
        image_id: int,
        c: typing.Optional[int],
):
    """ Takes a single channel image/mask and saves it as tiles.

    Args:
        image: 2d numpy array
        tile_stride: size of the output tiles
        out_dir: folder where the tiles will be saved
        image_id: duh
        c: 0/1/2 for intensity images. None for the mask.
    """
    if c is not None:
        if not (isinstance(c, int) and c in [0, 1, 2]):
            raise ValueError(f'`c` must be one of [None, 0, 1, 2]. Got {c} instead.')

    x_end, y_end = image.shape
    min_intensity = numpy.min(image)
    max_intensity = numpy.max(image)

    tile_template = numpy.zeros((tile_stride, tile_stride), dtype=numpy.float32)

    for x, x_min in enumerate(range(0, x_end, tile_stride)):
        x_max = min(x_end, x_min + tile_stride)

        for y, y_min in enumerate(range(0, y_end, tile_stride)):
            y_max = min(y_end, y_min + tile_stride)

            in_tile = image[x_min:x_max, y_min:y_max]
            if c is not None:
                in_tile = (in_tile - min_intensity) / (max_intensity - min_intensity + constants.EPSILON)

            out_tile = numpy.zeros_like(tile_template)
            out_tile[:x_max - x_min, :y_max - y_min] = in_tile[:]

            name = f'{image_id}_x{x:02d}_y{y:02d}' + ('' if c is None else f'_c{c:1d}') + '.npy'
            out_path = out_dir.joinpath(name)
            numpy.save(
                file=str(out_path),
                arr=out_tile,
                allow_pickle=False,
                fix_imports=False,
            )

    return


__all__ = [
    'tile_all_images',
    'resize_all_images',
]
