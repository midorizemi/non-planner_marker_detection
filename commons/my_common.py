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
import numpy as np
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
    def f(pair):
        p1=pair[0]
        p2=pair[1]
        return (p1.pt, p1.size, p1.angle, p1.response, p1.octave, p1.class_id), \
               (p2.pt, p2.size, p2.angle, p2.response, p2.octave, p2.class_id)
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
    def f(pair):
        p1=pair[0]
        p2=pair[1]
        return (p1.pt, p1.size, p1.angle, p1.response, p1.octave, p1.class_id),\
               (p2.pt, p2.size, p2.angle, p2.response, p2.octave, p2.class_id)
    index = tuple(tuple(f(pair)) for pairs in mesh_pairs for pair in pairs)
    return index

def load_pickle_matchepairs(fn):
    with open(fn, mode='rb') as f:
        index_pairs = pickle.load(f)

    pairs=list(list(cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                            _response=p[3], _octave=p[4], _class_id=p[5]) for p in ip) for ip in index_pairs)
    # pairs =[]
    # for i_pairs in index_pairs:
    #     p1 = i_pairs[0]
    #     p2 = i_pairs[1]
    #     pairs.append(
    #         [cv2.KeyPoint(x=p1[0][0], y=p1[0][1], _size=p1[1], _angle=p1[2],
    #                       _response=p1[3], _octave=p1[4], _class_id=p1[5]),
    #          cv2.KeyPoint(x=p2[0][0], y=p2[0][1], _size=p2[1], _angle=p2[2],
    #                       _response=p2[3], _octave=p2[4], _class_id=p2[5])]
    #     )
    return pairs

def load_pickle_mesh_matchepairs(fn, each_mesh_matchnum):
    from collections import deque
    with open(fn, mode='rb') as f:
        index_pairs = deque(pickle.load(f))

    # mesh_pairs = []
    # for matched_num in each_mesh_matchnum:
    #     for i in range(matched_num):
    #         i_pairs = index_pairs.popleft()
    #         p1 = i_pairs[0]
    #         p2 = i_pairs[1]
    #         pairs =[]
    #         pairs.append(
    #             [cv2.KeyPoint(x=p1[0][0], y=p1[0][1], _size=p1[1], _angle=p1[2],
    #                           _response=p1[3], _octave=p1[4], _class_id=p1[5]),
    #              cv2.KeyPoint(x=p2[0][0], y=p2[0][1], _size=p2[1], _angle=p2[2],
    #                           _response=p2[3], _octave=p2[4], _class_id=p2[5])]
    #         )
    #     mesh_pairs.append(pairs)

    def get_keypoint(i_pairs):
        p1 = i_pairs[0]
        p2 = i_pairs[1]
        return cv2.KeyPoint(x=p1[0][0], y=p1[0][1], _size=p1[1], _angle=p1[2],
                      _response=p1[3], _octave=p1[4], _class_id=p1[5]),\
               cv2.KeyPoint(x=p2[0][0], y=p2[0][1], _size=p2[1], _angle=p2[2],
                      _response=p2[3], _octave=p2[4], _class_id=p2[5])

    mesh_pairs = list(list(list(get_keypoint(index_pairs.popleft())) for i in range(matched_num)) for matched_num in each_mesh_matchnum )
    return mesh_pairs


def loader(which_one, fn, matchnum=None):
    if which_one is 'kd':
        #keypoint list and descriptor list
        return load_pikle(fn)
    if which_one is 'matched_pairs':
        #matched keypoint list
        return load_pickle_matchepairs(fn)
    if which_one is 'meshed_matched_pairs':
        #matched keypoint list
        return load_pickle_mesh_matchepairs(fn, matchnum)

def gamma_conversion(img):
    gamma = 2.0
    look_up_table = np.ones((256, 1), dtype='uint8') * 0
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img_gamma = cv2.LUT(img, look_up_table)

    return img_gamma