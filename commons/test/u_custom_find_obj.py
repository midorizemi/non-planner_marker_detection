"""
This is unit test split affine simulation sctipt
"""
import unittest
from unittest import TestCase
import cv2
import numpy as np
from copy import copy
from commons.common import Timer
from commons.find_obj import init_feature, filter_matches, explore_match
# import
import commons.affine_base as ab
from commons.custom_find_obj import filter_matches_wcross
import getopt
import sys

class TestClass(TestCase):
    def setUp(self):
        """ Set up befor test
        Using templates is qr.png
        Using input test image is 1.png
        Using feature function is SIFT
        Using split mesh number is 64
        """
        import sys, getopt
        self.opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
        self.opts = dict(self.opts)
        try:
            fn1, fn2 = args
        except:
            fn1 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'
            # fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/test/mltf_qrmarker/smpl_0.000000_0.000000.png'
            fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'

        self.img1 = cv2.imread(fn1, 0)
        self.img2 = cv2.imread(fn2, 0)
        self.splt_num = 64
        if self.img1 is None:
            print('Failed to load fn1:', fn1)
            sys.exit(1)

        if self.img2 is None:
            print('Failed to load fn2:', fn2)
            sys.exit(1)


    def test_cross_check(self):
        feature_name = self.opts.get('--feature', 'sift')
        detector, matcher = init_feature(feature_name)
        if detector is None:
            print('unknown feature:', feature_name)
            sys.exit(1)

        with Timer('affine simulation detecting'):
            kp1, desc1 = ab.affine_detect(detector, self.img1) #ASIFT
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = ab.affine_detect(detector, self.img2) #ASIFT
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches12 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2) #2
            raw_matches21 = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p2, p1, kp_pairs = filter_matches_wcross(kp1, kp2, raw_matches12, raw_matches21)
        print('matched points T=%d, Q=%d' % (len(p1), len(p2)))
        self.assertEqual(len(p1), len(p2), "Not same numbers")

    def test_cross_flann(self):
        feature_name = self.opts.get('--feature', 'sift-flann')
        detector, matcher = init_feature(feature_name)
        with Timer('affine simulation detecting'):
            kp1, desc1 = ab.affine_detect(detector, self.img1) #ASIFT
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = ab.affine_detect(detector, self.img2) #ASIFT
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches12 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2) #2
            raw_matches21 = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p2, p1, kp_pairs = filter_matches_wcross(kp1, kp2, raw_matches12, raw_matches21)
        print('matched points T=%d, Q=%d' % (len(p1), len(p2)))
        self.assertEqual(len(p1), len(p2), "Not same numbers")

    def test_cross_ORB(self):
        feature_name = self.opts.get('--feature', 'orb-flann')
        detector, matcher = init_feature(feature_name)
        with Timer('affine simulation detecting'):
            kp1, desc1 = ab.affine_detect(detector, self.img1) #ASIFT
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = ab.affine_detect(detector, self.img2) #ASIFT
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches12 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2) #2
            raw_matches21 = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p2, p1, kp_pairs = filter_matches_wcross(kp1, kp2, raw_matches12, raw_matches21)
        print('matched points T=%d, Q=%d' % (len(p1), len(p2)))
        self.assertEqual(len(p1), len(p2), "Not same numbers")

    @unittest.skip('Skip show images')
    def test_cross_check_dif(self):
        feature_name = self.opts.get('--feature', 'sift')
        detector, matcher = init_feature(feature_name)
        with Timer('affine simulation detecting'):
            kp1, desc1 = ab.affine_detect(detector, self.img1) #ASIFT
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = detector.detectAndCompute(self.img2, mask)
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches12 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2) #2
            raw_matches21 = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p2, p1, kp_pairs = filter_matches_wcross(kp1, kp2, raw_matches12, raw_matches21)
        print('matched points T=%d, Q=%d' % (len(p1), len(p2)))
        self.assertEqual(len(p1), len(p2), "Not same numbers")

    @unittest.skip('Skip show images')
    def test_cross_check_same(self):
        feature_name = self.opts.get('--feature', 'sift')
        detector, matcher = init_feature(feature_name)
        with Timer('Feature detecting'):
            h, w = self.img1.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp1, desc1 = detector.detectAndCompute(self.img1, mask)
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = detector.detectAndCompute(self.img2, mask)
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches12 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2) #2
            raw_matches21 = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p2, p1, kp_pairs = filter_matches_wcross(kp1, kp2, raw_matches12, raw_matches21)
        print('matched points T=%d, Q=%d' % (len(p1), len(p2)))
        self.assertEqual(len(p1), len(p2), "Not same numbers")
