#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample uses the affine transformation space sampling technique, called ASIFT [1],
but uses splited template base.
While the original implementation
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
from commons.affine_base import affine_detect, affine_skew, calc_affine_params
from commons.template_info import TemplateInfo as TmpInf
from commons.custom_find_obj import explore_match_for_meshes as show
from make_database import make_splitmap as mks
from make_database.split_affinesim import split_kd

def affine_detect_into_mesh(detector, split_num, img, mask=None, pool=None, simu_param = None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = calc_affine_params(simu_param)

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        s_kp, sdesc = split_kd(keypoints, descrs)
        return s_kp, sdesc

    splits_k = [[] for row in range(splt_num)]
    splits_d = [[] for row in range(splt_num)]
    keypoints, descrs = [], []
    if pool is None:
        ires = list(map(f, params))
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        for i, uk, ud in enumerate(k, d):
            splits_k.append(uk)
            splits_d.append(ud)

    return splits_k, splits_d

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2 = args
    except:
        fn1 = 'inputs/templates/qrmarker.png'
        fn2 = 'inputs/test/qrmarker.png'

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
    kp1, desc1 = affine_detect(detector, img1, pool=pool, simu_param='default')
    s_kp, s_desc = split_kd(kp1, desc1, splt_num)
    kp2, desc2 = affine_detect(detector, img2, pool=pool, simu_param='default')
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
            assert type(desc) == type(desc2), "EORROR TYPE"
            with Timer('matching'):
                raw_matches = matcher.knnMatch(desc2, trainDescriptors=desc, k=2)#2
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
        vis = show(win, img2, img1, list_kp_pairs, statuses, Hs)


    match_and_draw('affine find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()
