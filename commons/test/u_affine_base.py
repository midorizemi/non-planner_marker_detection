"""
This is unit test split affine simulation sctipt
"""
import unittest
from unittest import TestCase
import commons.affine_base
from commons.find_obj import init_feature
import cv2


@unittest.skip("SKIP")
class TestClass(TestCase):
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
            fn1 = 'inputs/templates/qr.png'
            fn2 = 'inputs/split_test/1.jpg'

        self.img1 = cv2.imread(fn1, 0)
        self.img2 = cv2.imread(fn2, 0)
        self.detector, self.matcher = init_feature(feature_name)
        self.splt_num = 64

    def test_affineskew(self):
        pass

    def test_a_detection(self):
        pass
