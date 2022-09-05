import pathlib

ROOT_DIR = (
    pathlib.Path(__file__)
    .parent.parent.parent.parent
)

INPUT_DIR = ROOT_DIR.joinpath('input', 'hubmap-organ-segmentation')

TRAIN_CSV = INPUT_DIR.joinpath('train.csv')
TEST_CSV = INPUT_DIR.joinpath('test.csv')
SAMPLE_SUBMISSION_CSV = INPUT_DIR.joinpath('sample_submission.csv')

TRAIN_IMAGES = INPUT_DIR.joinpath('train_images')
TRAIN_ANNOTATIONS = INPUT_DIR.joinpath('train_annotations')
TEST_IMAGES = INPUT_DIR.joinpath('test_images')

WORKING_DIR = ROOT_DIR.joinpath('working', 'hubmap-organ-segmentation')

SUBMISSION_PATH = WORKING_DIR.joinpath('submission.csv')

INITIAL_PATHS = [
    'INPUT_DIR',
    'TRAIN_CSV',
    'TEST_CSV',
    'SAMPLE_SUBMISSION_CSV',
    'TRAIN_IMAGES',
    'TRAIN_ANNOTATIONS',
    'TEST_IMAGES',
    'WORKING_DIR',
    'SUBMISSION_PATH',
]

TRAIN_TILES = WORKING_DIR.joinpath('train', 'intensity')
TRAIN_MASKS = WORKING_DIR.joinpath('train', 'labels')
