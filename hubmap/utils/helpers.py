import logging
import os
import random

import numpy

from . import constants


def make_logger(name: str, level: str = None):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.KAGGLE_LOG if level is None else level)
    return logger_


logger = make_logger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    return


def rle_to_mask(rle: str, height: int, width: int):
    """ Converts a run-length encoded string into a 2d mask.

    Args:
        rle: Run-Length Encoded string.
        height: of the mask
        width: of the mask

    Returns:
        2d mask
    """
    s = rle.split()
    start, length = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    mask = numpy.zeros(height * width, dtype=numpy.float32)
    for i, l in zip(start, length):
        mask[i:i + l] = 1
    mask = mask.reshape(width, height).T
    mask = numpy.ascontiguousarray(mask)
    return mask


def mask_to_rle(mask: numpy.ndarray):
    """ Converts a mask to a run-length encoded string.

    Args:
        mask: A 2d segmentation mask.

    Returns:
        A run-length encoded string.
    """
    assert mask.ndim == 2, f'`mask` must be a 2d array. Got {mask.ndim} dimensions instead ...'

    m = mask.T.flatten()
    m = numpy.concatenate([[0], m, [0]])
    run = numpy.where(m[1:] != m[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    return rle


__all__ = [
    'make_logger',
    'seed_everything',
    'rle_to_mask',
    'mask_to_rle',
]
