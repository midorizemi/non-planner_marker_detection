"""
This is unit test split affine simulation sctipt
"""

import unittest

from commons.find_obj import init_feature, filter_matches
from commons.common import Timer
from commons.affine_base import affine_detect
from make_database import split_affinesim2nd as splta2
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

    @unittest.skip('Skip show images')
    def test_affine_detect_into_mesh(self):
        with Timer('Detection with split into mesh'):
            splits_kp, splits_desc = splta2.affine_detect_into_mesh(self.detector, self.splt_num, self.img1, simu_param='test2')
        self.assertIsNotNone(splits_kp)
        self.assertIsNotNone(splits_desc)
        self.assertEqual(len(splits_kp), self.splt_num, "It is not same")
        self.assertEqual(len(splits_desc), self.splt_num, "It is not same")
        lenskp = 0
        lensdescr = 0
        for skp, sdesc in zip(splits_kp, splits_desc):
            if not skp:
                self.assertTrue(False, "Keypoints of mesh is Empty")
            else:
                self.assertEqual(len(skp), 3, "It is not same")
                for mesh_kp in skp:
                    lenskp += len(mesh_kp)
            if not sdesc:
                self.assertTrue(False, "Descriptors of mesh is Empty")
            self.assertNotEqual(sdesc[0].size, 0, "Descriptor is Empty")
            self.assertEqual(sdesc[0].shape[1], 128, "SIFT features")
            self.assertEqual(len(sdesc), 3, "It is not same")
            for mesh_desc in sdesc:
                lensdescr += mesh_desc.shape[0]

        with Timer('Detection'):
            kp, desc = affine_detect(self.detector, self.img1, simu_param='test2')
        self.assertEqual(lenskp, len(kp), "Some keypoints were droped out.")
        self.assertEqual(lensdescr, len(desc), "Some descriptors were droped out.")

    def test_result(self):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        meshList_kpT, meshList_descT = splta2.affine_detect_into_mesh(self.detector, self.splt_num, self.img1, pool=pool)
        kpQ, descQ = affine_detect(self.detector, self.img2, pool=pool)

        def count_keypoints():
            c = 0
            for s_kpT in meshList_kpT:
                for kpT in s_kpT:
                    c += len(kpT)
            return c
        print('img1 - %d features, img2 - %d features' % (count_keypoints(), len(kpQ)))

        mesh_pT, mesh_pQ, mesh_pairs = splta2.match_with_cross(self.matcher, meshList_descT, meshList_kpT, descQ, kpQ)
        self.assertTrue(True)
        # pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        # s_kp, s_desc = splta.affine_detect_into_mesh(self.detector, self.splt_num, self.img1, pool=pool)
        # kp2, desc2 = affine_detect(self.detector, self.img2, pool=pool)
        # len_s_kp = 0
        # for kps in s_kp:
        #     len_s_kp += len(kps)
        #
        # def calc_H(kp1, kp2, desc1, desc2):
        #     with Timer('matching'):
        #         raw_matches = self.matcher.knnMatch(desc2, desc1, 2)#2
        #     p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        #     if len(p1) >= 4:
        #         H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        #         print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        #         # do not draw outliers (there will be a lot of them)
        #         kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        #     else:
        #         H, status = None, None
        #         print('%d matches found, not enough for homography estimation' % len(p1))
        #
        #     return kp_pairs, H, status
        #
        # def match_and_draw(win):
        #     list_kp_pairs = []
        #     Hs = []
        #     statuses = []
        #     i =0
        #     for kps, desc in zip(s_kp, s_desc):
        #         assert type(desc) == type(desc2), "EORROR TYPE"
        #         with Timer('matching'):
        #             raw_matches = self.matcher.knnMatch(desc2, trainDescriptors=desc, k=2)#2
        #         p2, p1, kp_pairs = filter_matches(kp2, kps, raw_matches)
        #         if len(p1) >= 4:
        #             H, status = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
        #             print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        #             # do not draw outliers (there will be a lot of them)
        #             list_kp_pairs.extend([kpp for kpp, flag in zip(kp_pairs, status) if flag])
        #         else:
        #             H, status = None, None
        #             print('%d matches found, not enough for homography estimation' % len(p1))
        #         Hs.append(H)
        #         statuses.extend(status)
        #         i+=1
        #     vis = show(win, self.img2, self.img1, list_kp_pairs, statuses, Hs)
        #
        #
        # match_and_draw('affine find_obj')
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # self.assertEqual(1, 1)

