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
from commons.find_obj import init_feature, filter_matches
from commons.affine_base import affine_detect, affine_skew, calc_affine_params
from commons.custom_find_obj import explore_match_for_meshes as show, filter_matches_wcross as c_filter
from commons.custom_find_obj import calclate_Homography, explore_match_for_meshes
from make_database.split_affinesim import split_kd

def affine_detect_into_mesh(detector, split_num, img, mask=None, pool=None, simu_param=None):
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
        timg, tmask, Ai = affine_skew(t, phi, img, mask)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        s_kp, sdesc = split_kd(keypoints, descrs, split_num)
        return s_kp, sdesc

    splits_k = [[] for row in range(split_num)]
    splits_d = [[] for row in range(split_num)]
    if pool is None:
        ires = list(map(f, params))
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        for j, uk in enumerate(k):
            splits_k[j].append(uk)
            splits_d[j].append(d[j])

    return splits_k, splits_d

def match_with_cross(matcher, mmeshList_dsscT, meshList_kpT, descQ, kpQ):
    """
   You have to input mmeshList_dsscT is affine_detect_into_mesh
    :param matcher:
    :param mmeshList_dsscT: mesh-split descts
    :param meshList_kpT:
    :param descQ:
    :param kpQ:
    :return:
    """
    max_pairs = -1
    mesh_pT = []
    mesh_pQ =[]
    mesh_pairs = []
    for mesh_kpT, mesh_descT in zip(meshList_kpT, mmeshList_dsscT):
        """メッシュ毎のキーポイント，特徴量"""
        a = []
        b = []
        c = []
        for kpT, descT in zip(mesh_kpT, mesh_descT):
            """パラメータ毎に得られるキーポイント，特徴量"""
            raw_matchesTQ = matcher.knnMatch(descQ, trainDescriptors=descT, k=2)
            raw_matchesQT = matcher.knnMatch(descT, trainDescriptors=descQ, k=2)
            result = c_filter(kpT, kpQ, raw_matchesTQ, raw_matchesQT)
            if max_pairs <= len(result[2]):
                max_pairs = len(result[2])
                a = result[0]
                b = result[1]
                c = result[2]
        mesh_pT.append(a)
        mesh_pQ.append(b)
        mesh_pairs.append(c)
    return mesh_pT, mesh_pQ, mesh_pairs

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2 = args
    except:
        fn1 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/templates/qrmarker.png'
        fn2 = '/home/tiwasaki/PycharmProjects/makeDB/inputs/test/mltf_qrmarker/smpl_1.414214_152.735065.png'

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
    with Timer('Detection affine simulation Ver.2'):
        s_kp, s_desc = affine_detect_into_mesh(detector, splt_num, img1, pool=pool, simu_param='default')
    kp2, desc2 = affine_detect(detector, img2, pool=pool, simu_param='test')

    def count_keypoints():
        c = 0
        for s_kpT in s_kp:
            for kpT in s_kpT:
                c += len(kpT)
        return c
    print('img1 - %d features, img2 - %d features' % (count_keypoints(), len(kp2)))

    with Timer('matching'):
        mesh_pT, mesh_pQ, mesh_pairs = match_with_cross(matcher, s_desc, s_kp, desc2, kp2)
    Hs = []
    statuses = []
    kp_pairs_long = []
    for pT, pQ, pairs in zip(mesh_pT, mesh_pQ, mesh_pairs):
        pairs, H, status = calclate_Homography(pT, pQ, pairs)
        Hs.append(H)
        statuses.append(status)
        for p in pairs:
            kp_pairs_long.append(p)

    viw = explore_match_for_meshes('affine find_obj', img1, img2, kp_pairs_long, Hs=Hs)
    cv2.waitKey()
    cv2.destroyAllWindows()
