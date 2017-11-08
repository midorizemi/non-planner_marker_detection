"""
This is unit test split affine simulation sctipt
"""

import unittest

import itertools as it
from multiprocessing.pool import ThreadPool

from commons.find_obj import init_feature, filter_matches
from commons.common import Timer
from make_database import split_affinesim as splta

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
            fn1 = '~/PycharmProjects/makeDB/inputs/templates/qrmarker.png'
            fn2 = '~/PycharmProjects/makeDB/inputs/test/qrmarker.png'

        self.img1 = cv2.imread(fn1, 0)
        self.img2 = cv2.imread(fn2, 0)
        self.detector, self.matcher = init_feature(feature_name)
        self.splt_num = 64

    def test_a(self):
        cv2.imshow('test', self.img1)
        cv2.waitKey()
        self.assertIsNotNone(self.img1, "画像がありません")
        self.assertIsNotNone(self.img2, "画像がありません")
        self.assertIsNotNone(self.img2, "無効な画像特徴名を指定しています")

    def test_splitkd(self):
        self.assertIsNotNone(self.img1, "画像がありません")
        self.assertIsNotNone(self.img2, "画像がありません")
        self.assertIsNotNone(self.img2, "無効な画像特徴名を指定しています")
        kp, desc = splta.affine_detect(self.detector, self.splt_num, self.img1)
        s_kp, s_desc = splta.split_kd(kp, desc, self.splt_num)
        self.assertIsNotNone(s_kp)
        self.assertIsNotNone(s_desc)
        self.assertEqual(len(s_kp), self.split_num)
        self.assertEqual(len(s_desc), self.split_num)
        for i, kp in enumerate(s_kp):
            if not kp:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
            if not s_desc[i]:
                self.assertTrue(True)
            else:
                self.assertTrue(False)

    @unittest.skip("SKIP")
    def test_split_affine(self):
        s_kp, s_desc = splta.affine_detect_into_mesh(self.detector, self.splt_num, self.img1)
        self.assertEqual(len(s_kp), self.split_num)
        self.assertEqual(len(s_desc), self.split_num)

    @unittest.skip("SKIP")
    def test_result(self):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp1, desc1 = splta.affine_detect(self.detector, self.img1, pool=pool)
        s_kp, s_desc = splta.split_kd(kp1, desc1, self.splt_num)
        kp2, desc2 = splta.affine_detect(self.detector, self.img2, pool=pool)
        len_s_kp = 0
        for kps in s_kp:
            len_s_kp += len(kps)
        print('img1 - %d features, img2 - %d features' % (len_s_kp, len(kp2)))

        def calc_H(kp1, kp2, desc1, desc2):
            with Timer('matching'):
                raw_matches = splta.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)#2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
            if len(p1) >= 4:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                # do not draw outliers (there will be a lot of them)
                kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
            else:
                H, status = None, None
                print('%d matches found, not enough for homography estimation' % len(p1))

            return kp_pairs, H, status

        def match_and_draw(win):
            list_kp_pairs = []
            Hs = []
            statuses = []
            for i, kps in enumerate(s_kp):
                kp_pairs, Hs[i], status = calc_H(kps, kp2, s_desc[i], desc2)
                list_kp_pairs.extend(kp_pairs)
                statuses.extend(status)
            vis = self.show(win, self.img1, self.img2, list_kp_pairs, statuses, Hs)


        match_and_draw('affine find_obj')
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.assertEqual(1, 1)

