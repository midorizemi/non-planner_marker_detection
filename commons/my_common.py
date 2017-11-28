from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2

# built-in modules
import os
import itertools as it
from contextlib import contextmanager
from commons.common import clock

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

@contextmanager
def Timer(msg, logger):
    logger.info(msg+'...')
    start = clock()
    try:
        yield
    finally:
        logger.info("%.2f ms\n" % ((clock()-start)*1000))
