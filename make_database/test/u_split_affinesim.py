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
            import os
            dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
            fn1 = os.path.abspath(os.path.join(dir, 'data/templates/qrmarker.png'))
            fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))

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

    def test_splitkd(self):
        kp, desc = affine_detect(self.detector, self.img1, None, None, simu_param='test2')
        s_kp, s_desc = splta.split_kd(kp, desc, self.splt_num)
        self.assertIsNotNone(s_kp)
        self.assertIsNotNone(s_desc)
        self.assertEqual(len(s_kp), 64)
        self.assertEqual(len(s_desc), 64)
        lenskp = 0
        lensdescr = 0
        for skp, sdesc in zip(s_kp, s_desc):
            if not skp:
                self.assertTrue(False, "Keypoints is Empty")
            else:
                lenskp += len(skp)
                self.assertTrue(True)
            self.assertNotEqual(sdesc.size, 0, "Descriptor is Empty")
            lensdescr += sdesc.shape[0]
        print("{0} == {1}".format(lenskp, len(kp)))
        self.assertEqual(lenskp, len(kp), "Some keypoints were droped out.")
        self.assertEqual(lensdescr, len(desc), "Some descriptors were droped out.")

    @unittest.skip('Skip show images')
    def test_result(self):
        s_kp, s_desc = splta.affine_detect_into_mesh(self.detector, self.splt_num, self.img1)
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp2, desc2 = affine_detect(self.detector, self.img2, pool=pool)
        len_s_kp = 0
        for kps in s_kp:
            len_s_kp += len(kps)
        print('imgQ - %d features, imgT - %d features' % (len_s_kp, len(kp2)))

        def calc_H(kp1, kp2, desc1, desc2):
            with Timer('matching'):
                raw_matches = self.matcher.knnMatch(desc2, desc1, 2)#2
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
            i =0
            for kps, desc in zip(s_kp, s_desc):
                assert type(desc) == type(desc2), "EORROR TYPE"
                with Timer('matching'):
                    raw_matches = self.matcher.knnMatch(desc2, trainDescriptors=desc, k=2)#2
                p2, p1, kp_pairs = filter_matches(kp2, kps, raw_matches)
                if len(p1) >= 4:
                    H, status = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
                    print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                    # do not draw outliers (there will be a lot of them)
                    list_kp_pairs.extend([kpp for kpp, flag in zip(kp_pairs, status) if flag])
                else:
                    H, status = None, None
                    print('%d matches found, not enough for homography estimation' % len(p1))
                Hs.append(H)
                statuses.extend(status)
                i+=1
            vis = show(win, self.img2, self.img1, list_kp_pairs, statuses, Hs)


        match_and_draw('affine find_obj')
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.assertEqual(1, 1)

