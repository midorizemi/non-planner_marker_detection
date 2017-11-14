"""
This is unit test split affine simulation sctipt
"""

import unittest

from commons.find_obj import init_feature, filter_matches
from commons.common import Timer
from commons.affine_base import affine_detect
from make_database import split_affinesim as splta
from commons.custom_find_obj import explore_match_for_meshes as show

import cv2
import numpy as np

# built-in modules
from multiprocessing.pool import ThreadPool

class TestSplitAffineSim(unittest.TestCase):

    def setUp(self):
        """ Set up befor test
        Using templates is qr.png
        Using input test image is 1.png
        Using feature function is SIFT
        Using split mesh number is 64
        """
        import sys, getopt
        opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
        opts = dict(opts)
        feature_name = opts.get('--feature', 'sift')
        try:
            fn1, fn2 = args
        except:
            fn1 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'
            fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/test/mltf_qrmarker/smpl_0.000000_0.000000.png'

        self.img1 = cv2.imread(fn1, 0)
        self.img2 = cv2.imread(fn2, 0)
        self.detector, self.matcher = init_feature(feature_name)
        self.splt_num = 64
        if self.img1 is None:
            print('Failed to load fn1:', fn1)
            sys.exit(1)

        if self.img2 is None:
            print('Failed to load fn2:', fn2)
            sys.exit(1)

        if self.detector is None:
            print('unknown feature:', feature_name)
            sys.exit(1)
