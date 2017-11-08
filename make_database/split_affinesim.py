#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

# local modules
from commons.common import Timer
from commons.find_obj import init_feature, filter_matches, explore_match
from commons.affine_base import affine_detect
from commons.template_info import TemplateInfo as TmpInf
from commons.custom_find_obj import explore_match_for_meshes as show
from make_database import make_splitmap as mks

def split_kd(keypoints, descrs, splt_num):
    tmp = TmpInf()
    split_tmp_img = mks.make_splitmap(tmp)
    assert isinstance(split_tmp_img, np.ndarray)
    global descrs_list
    if isinstance(descrs, np.ndarray):
        descrs_list = descrs.tolist()
    splits_k = [[] for row in range(splt_num)]
    splits_d = [[] for row in range(splt_num)]

    assert isinstance(keypoints, list)
    assert isinstance(descrs_list, list)
    for keypoint, descr in zip(keypoints, descrs_list):
        x, y = np.int32(keypoint.pt)
        if x < 0 or x >= 800 or y < 0 or y >= 600:
            continue
        splits_k[split_tmp_img[y, x][0]].append(keypoint)
        splits_d[split_tmp_img[y, x][0]].extend(descr)

    for i, split_d in enumerate(splits_d):
        splits_d[i] = np.array(split_d)

    return splits_k, splits_d

def affine_detect_into_mesh(detector, split_num, img1, pool):
    kp, desc = affine_detect(detector, img1, pool=pool)
    return split_kd(kp, desc, split_num)

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2 = args
    except:
        fn1 = 'inputs/test/aero1.jpg'
        fn2 = 'inputs/test/aero3.jpg'

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    splt_num = 64
    print('using', feature_name)
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    s_kp, s_desc = split_kd(kp1, desc1, splt_num)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    len_s_kp = 0
    for kps in s_kp:
        len_s_kp += len(kps)
    print('img1 - %d features, img2 - %d features' % (len_s_kp, len(kp2)))

    def calc_H(kp1, kp2, desc1, desc2):
        with Timer('matching'):
            raw_matches = matcher.knnMatch(desc2, desc1, 2)#2
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
            asset isinstance(desc, type(desc2))
            with Timer('matching'):
                raw_matches = matcher.knnMatch(desc2, trainDescriptors=desc, k=2)#2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
            if len(p1) >= 4:
                Hs[i], status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                # do not draw outliers (there will be a lot of them)
                list_kp_pairs.extend([kpp for kpp, flag in zip(kp_pairs, status) if flag])
            else:
                Hs[i], status = None, None
                print('%d matches found, not enough for homography estimation' % len(p1))
            statuses.extend(status)
            i+=1
        vis = show(win, img1, img2, list_kp_pairs, statuses, Hs)


    match_and_draw('affine find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()

