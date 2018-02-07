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
        logger.info("%.2f ms\n" % ((clock() - start) * 1000))


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)


def load_pikle(fn):
    with open(fn, mode='rb') as f:
        index, des = pickle.load(f)
    kp = []
    for p in index:
        temp = cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                            _response=p[3], _octave=p[4], _class_id=p[5])
        kp.append(temp)

    return kp, des


def format4pickle_kp(mesh_kp):
    ##キーポイントを出力するための整形処理
    index = []
    for kp in mesh_kp:
        sub_index = []
        for p in kp:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            sub_index.append(temp)
        index.append(sub_index)
    return index


def format4pickle_pairs(mesh_pairs):
    ##キーポイントを出力するための整形処理
    # index = []
    # for pairs in mesh_pairs:
    #     sub_index = []
    #     for pair in pairs:
    #         temp = ((p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in pair)
    #         sub_index.append(temp)
    #     index.append(sub_index)


    index = tuple(
        tuple((p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in pair) for pairs in mesh_pairs for pair
        in pairs)
    return index
