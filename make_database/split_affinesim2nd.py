#!/usr/bin/env python

'''
USAGE
    使わない!
  split_affinesim2nd.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

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
from commons.custom_find_obj import filter_matches_wcross as c_filter, explore_match_for_meshes
from commons.custom_find_obj import calclate_Homography, calclate_Homography_hard, draw_matches_for_meshes
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

def match_with_cross(matcher, meshList_descQ, meshList_kpQ, descT, kpT):
    """
   You have to input mmeshList_dsscT is affine_detect_into_mesh
    :param matcher:
    :param meshList_descQ: mesh-split descts
    :param meshList_kpQ:
    :param descT:
    :param kpT:
    :return:
    """
    max_pairs = -1
    mesh_pT = []
    mesh_pQ =[]
    mesh_pairs = []
    for mesh_kpQ, mesh_descQ in zip(meshList_kpQ, meshList_descQ):
        """メッシュ毎のキーポイント，特徴量"""
        q = []
        t = []
        pr = []
        for kpQ, descQ in zip(mesh_kpQ, mesh_descQ):
            """パラメータ毎に得られるキーポイント，特徴量"""
            raw_matchesTQ = matcher.knnMatch(descT, trainDescriptors=descQ, k=2)
            raw_matchesQT = matcher.knnMatch(descQ, trainDescriptors=descT, k=2)
            result = c_filter(kpQ, kpT, raw_matchesQT, raw_matchesTQ)
            if max_pairs <= len(result[2]):
                max_pairs = len(result[2])
                q = result[0]
                t = result[1]
                pr = result[2]
        mesh_pQ.append(q)
        mesh_pT.append(t)
        mesh_pairs.append(pr)
    return mesh_pQ, mesh_pT, mesh_pairs

def count_keypoints(s_kp):
    length = 0
    for skp in s_kp:
        for mesh_kp in skp:
            length += len(mesh_kp)
    return length
if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2 = args
    except:
        import os
        dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        fn1 = os.path.abspath(os.path.join(dir, 'data/templates/qrmarker.png'))
        fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))

    imgQ = cv2.imread(fn1, 0)
    imgT = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature_name)

    if imgQ is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if imgT is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    splt_num = 64
    print('using', feature_name)
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection affine simulation Ver.2'):
        s_kpQ, s_descQ = affine_detect_into_mesh(detector, splt_num, imgQ, pool=pool, simu_param='default')
    kpT, descT = affine_detect(detector, imgT, pool=pool, simu_param='test')

    print('imgQ - %d features, imgT - %d features' % (count_keypoints(s_kpQ), len(kpT)))

    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = match_with_cross(matcher, s_descQ, s_kpQ, descT, kpT)

    Hs = []
    statuses = []
    kp_pairs_long = []
    Hs_stable = []
    kp_pairs_long_stable = []
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        pairs, H, status = calclate_Homography(pQ, pT, pairs)
        Hs.append(H)
        statuses.append(status)
        if status is not None and np.sum(status)/len(status) >= 0.4:
            Hs_stable.append(H)
        else:
            Hs_stable.append(None)
        for p in pairs:
            kp_pairs_long.append(p)
            if np.sum(status)/len(status) >= 0.4:
                kp_pairs_long_stable.append(p)

    vis = draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    cv2.imshow('view weak meshes', vis)
    cv2.imwrite('qr2_meshes.png', vis)

    visS = draw_matches_for_meshes(imgQ, imgT, Hs=Hs_stable)
    cv2.imshow('view stable meshes', visS)
    cv2.imwrite('qr2_meshes_stable.png', visS)

    viw = explore_match_for_meshes('affine find_obj', imgQ, imgT, kp_pairs_long_stable, Hs=Hs_stable)
    cv2.imwrite('qr2_mesh_line.png', viw)
    cv2.waitKey()
    cv2.destroyAllWindows()

