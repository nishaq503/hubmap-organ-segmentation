import inspect
import logging
import pathlib

from hubmap.utils import helpers
from hubmap.utils import paths

#delete this later

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger('hubmap.main')


logger.info('Hello from hubmap!')


for _, path in inspect.getmembers(paths):
    if isinstance(path, pathlib.PosixPath):
        if path.exists(): 
            logger.info(f'Found path: {path}')
        else: 
            logger.warn(f'Path not found: {path}')
            break
else: 
    logger.info('Found all paths')
