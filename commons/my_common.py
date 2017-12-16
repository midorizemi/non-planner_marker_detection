from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2

# built-in modules
from contextlib import contextmanager
from commons.common import clock
from logging import getLogger

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']
logger = getLogger(__name__)

@contextmanager
def Timer(msg):
    logger.info(msg+'...')
    start = clock()
    try:
        yield
    finally:
        logger.info("%.2f ms\n" % ((clock()-start)*1000))
