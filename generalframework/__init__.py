name = "generalframework"

import logging
import os
import sys
from enum import Enum

LOGGER = logging.getLogger(__name__)
LOGGER.parent = None


class ModelMode(Enum):
    """ Different mode of model """
    TRAIN = 'TRAIN'  # during training
    EVAL = 'EVAL'  # eval mode. On validation data
    PRED = 'PRED'

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == 'train':
            return ModelMode.TRAIN
        elif mode_str == 'eval':
            return ModelMode.EVAL
        elif mode_str == 'predict':
            return ModelMode.PRED
        else:
            raise ValueError('Invalid argument mode_str {}'.format(mode_str))


def config_logger(log_dir):
    """ Get console handler """
    log_format = logging.Formatter("[%(module)s - %(asctime)s - %(levelname)s] %(message)s")
    LOGGER.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_format)

    fh = logging.FileHandler(os.path.join(log_dir, 'log.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(log_format)

    LOGGER.handlers = [console_handler, fh]
