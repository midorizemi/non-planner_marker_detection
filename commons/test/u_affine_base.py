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
from commons.find_obj import init_feature

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
            fn1 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'
            # fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/test/mltf_qrmarker/smpl_0.000000_0.000000.png'
            fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'

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

    @unittest.skip('Skip show images')
    def test_affineskew(self):
        pass

    def test_affine_detection(self):
        with Timer('affine simulation detecting'):
            kp1, desc1 = ab.affine_detect(self.detector, self.img1) #ASIFT
        with Timer('Feature detecting'):
            h, w = self.img2.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
            kp2, desc2 = self.detector.detectAndCompute(self.img2, mask)
            # kp2, desc2 = ab.affine_detect(self.detector, self.img2, simu_param='test') #SIFT
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        with Timer('matching'):
            raw_matches = self.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        self.assertEqual(len(kp_pairs), len(kp2), "kp_pairs and sift_keypoints are not equal")
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))
        vis = explore_match('test_affine_detection', self.img1, self.img2, kp_pairs, None, H)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print('Ckeck SIFT matching numbers')
        kp3 = copy(kp2)
        desc3 = np.copy(desc2)
        raw_matches_ = self.matcher.knnMatch(desc3, trainDescriptors=desc2, k=2) #2
        p3, p2_, kp_pairs_ = filter_matches(kp3, kp2, raw_matches_)
        self.assertEqual(len(kp_pairs), len(kp_pairs_), "keypoints pairs and sift_keypoints_pairs is not equal")
