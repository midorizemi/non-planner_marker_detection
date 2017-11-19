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
from commons.find_obj import init_feature
from commons.affine_base import affine_detect
from commons.template_info import TemplateInfo as TmpInf
from commons.custom_find_obj import explore_match_for_meshes, filter_matches_wcross as c_filter
from commons.custom_find_obj import calclate_Homography
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
        if x < 0 or x >= 800:
            if x < 0:
                x = 0
            else:
                x = 799
        elif y < 0 or y >= 600:
            if y < 0:
                y = 0
            else:
                y = 599
        splits_k[split_tmp_img[y, x][0]].append(keypoint)
        splits_d[split_tmp_img[y, x][0]].append(descr)

    for i, split_d in enumerate(splits_d):
        splits_d[i] = np.array(split_d, dtype=np.float32)

    return splits_k, splits_d

def affine_detect_into_mesh(detector, split_num, img1, mask=None, pool=None):
    kp, desc = affine_detect(detector, img1, mask, pool=pool)
    return split_kd(kp, desc, split_num)

def match_with_cross(matcher, meshList_descT, meshList_kpT, descQ, kpQ):
    meshList_pT = []
    meshList_pQ = []
    meshList_pairs = []
    for mesh_kpT, mesh_descT in zip(meshList_kpT, meshList_descT):
        raw_matchesTQ = matcher.knnMatch(descQ, trainDescriptors=mesh_descT, k=2)
        raw_matchesQT = matcher.knnMatch(mesh_descT, trainDescriptors=descQ, k=2)
        pT, pQ, pairs = c_filter(mesh_kpT, kpQ, raw_matchesTQ, raw_matchesQT)
        meshList_pT.append(pT)
        meshList_pQ.append(pQ)
        meshList_pairs.append(pairs)
    return meshList_pT, meshList_pQ, meshList_pairs

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
    kp1, desc1 = affine_detect(detector, img1, pool=pool, simu_param='default')
    s_kp, s_desc = split_kd(kp1, desc1, splt_num)
    kp2, desc2 = affine_detect(detector, img2, pool=pool, simu_param='test')
    len_s_kp = 0
    for kps in s_kp:
        len_s_kp += len(kps)
    print('img1 - %d features, img2 - %d features' % (len_s_kp, len(kp2)))

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

