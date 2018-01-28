from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce
# built-in modules
from contextlib import contextmanager
from commons.common import clock
from logging import getLogger
import cv2
import os
import pickle

logger = getLogger(__name__)

@contextmanager
def Timer(msg):
    logger.info('Measuring Time {}'.format(msg))
    start = clock()
    try:
        yield
    finally:
        logger.info("%.2f ms\n" % ((clock()-start)*1000))

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)

def get_pikle(*args, **kwargs):
    return os.path.join(kwargs.get('base_name', ''), args, kwargs.get('fn', 'example') + 'pikle')

def load_pikle(fn):
    with open(fn, mode='rb') as f:
        index, des = pickle.load(f)
    kp = []
    for p in index:
        temp = cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                            _response=p[3], _octave=p[4], _class_id=p[5])
        kp.append(temp)