#!/usr/bin/env python

'''
USAGE
  split_affinesim.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

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
from commons.custom_find_obj import calclate_Homography, calclate_Homography_hard, draw_matches_for_meshes
from make_database import make_splitmap as mks
from commons import my_file_path_manager as myfm
import make_database.split_affinesim as splaf

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
        if y < 0 or y >= 600:
            if y < 0:
                y = 0
            else:
                y = 599
        splits_k[split_tmp_img[y, x][0]].append(keypoint)
        splits_d[split_tmp_img[y, x][0]].append(descr)

    for i, split_d in enumerate(splits_d):
        splits_d[i] = np.array(split_d, dtype=np.float32)

    return splits_k, splits_d

def merge_rule(splt_k: list, splt_d: list, temp_inf: TmpInf):
    """
    何かしらのマージルール
    特徴点数とか，分布とか，特徴量とかでマージが必要なメッシュかをかをはんていする
    """
    mesh_k_num = np.array([len(keypoints) for keypoints in splt_k]).reshape(temp_inf.get_mesh_shape())

def analysis_num(mesh_k_num):
    #分析１：特徴点数のバラつき
    mean = mesh_k_num.mean()
    median = np.median(mesh_k_num)
    max = np.amax(mesh_k_num)
    min = np.amax(mesh_k_num)
    peak2peak = np.ptp(mesh_k_num)
    standard_deviation = np.std(mesh_k_num)
    variance = np.var(mesh_k_num)
    return mean, median, max, min, peak2peak, standard_deviation, variance

from matplotlib.axes import Axes
import pandas as pd
from typing import Tuple
def analysis_kp(splt_k, temp_inf: TmpInf) -> Tuple[Axes, pd.DataFrame]:
    """
    #分析2：特徴点座標のバラつき
    :param splt_k:
    :param temp_inf:
    :return:
    """
    import seaborn as sns
    mesh_k_np = [[np.int32(i), np.int32(kp.pt[0]), np.int32(kp.pt[1])] for i, keypoints in enumerate(splt_k)
        for kp in keypoints]
    df = pd.DataFrame(mesh_k_np, columns=['mesh_id', 'x', 'y'])
    print("Done make data")
    print(df.head(5))
    with Timer('plotting Kernel De'):
        for i in range(temp_inf.get_splitnum()):
            ax = sns.kdeplot(df.query('mesh_id == ' + str(i))['x'], df.query('mesh_id == ' + str(i))['y'], shade=True)
            ax.set(ylim=(600, 0))
            ax.set(xlim=(0, 800))
            ax.set(xlabel="x")
            ax.set(ylabel="y")
            ax.set(title="Kernel density estimation")

    # ax = sns.kdeplot(df['x'], df['y'], shade=True)
    return ax, df

def combine_mesh(split_k, split_d, temp_inf):
    """
    :type temp_inf: TmpInf
    :param split_k:
    :param split_d:
    :param temp_inf:
    :return:
    """
    mesh_map = temp_inf.get_mesh_map()
    mesh_k_num = np.array(len(keypoints) for keypoints in split_k).reshape(temp_inf.get_mesh_shape())

    for i, kd in enumerate(zip(split_k, split_d)):
        """
        矩形メッシュをマージする．4近傍のマージ．順番は左，上，右，下．
        最大値のところとマージする
        :return:
        """
        if merge_rule() is True:
            meshid_list = temp_inf.get_meshidlist_nneighbor(i)
            self_id = mesh_map[temp_inf.get_meshid_vertex(i)]
            if not self_id == i or len(np.where(mesh_map == i)[0]) > 1:
                """すでにマージされている"""
                continue
            #最大値のindexを求める．
            dtype = [('muki', int), ('keypoint_num', int), ('merge_id', int)]
            tmp = np.array([(len(meshid_list)-index, mesh_k_num[id], id) for index, id in enumerate(meshid_list)],
                           dtype=dtype)
            tmp.sort(order=['keypoint_num', 'muki'])
            #idにself_idをマージする
            merge_id = tmp[-1][2]
            mesh_map[temp_inf.get_meshid_vertex(i)] = merge_id
            split_k[merge_id].extend(split_k[i])
            np.concatenate((split_d[merge_id], split_d[i]))
            #マージされて要らなくなったメッシュは消す
            split_k[i] = []
            split_d[i] = np.array([[]])


def affine_detect_into_mesh(detector, split_num, img1, mask=None, simu_param='default'):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp, desc = affine_detect(detector, img1, mask, pool=pool, simu_param=simu_param)
    return split_kd(kp, desc, split_num)

def match_with_cross(matcher, meshList_descQ, meshList_kpQ, descT, kpT):
    meshList_pQ = []
    meshList_pT = []
    meshList_pairs = []
    for mesh_kpQ, mesh_descQ in zip(meshList_kpQ, meshList_descQ):
        raw_matchesQT = matcher.knnMatch(mesh_descQ, trainDescriptors=descT, k=2)
        raw_matchesTQ = matcher.knnMatch(descT, trainDescriptors=mesh_descQ, k=2)
        pQ, pT, pairs = c_filter(mesh_kpQ, kpT, raw_matchesQT, raw_matchesTQ)
        meshList_pT.append(pT)
        meshList_pQ.append(pQ)
        meshList_pairs.append(pairs)
    return meshList_pQ, meshList_pT, meshList_pairs

def count_keypoints(splt_kpQ):
    len_s_kp = 0
    for kps in splt_kpQ:
        len_s_kp += len(kps)
    return len_s_kp
def test_module():
    import os
    import sys
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    fn1 = os.path.abspath(os.path.join(dir, 'data/templates/qrmarker.png'))
    fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))
    imgQ = cv2.imread(fn1, 0)
    imgT = cv2.imread(fn2, 0)
    detector, matcher = init_feature('sift')
    if imgQ is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if imgT is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    template_information = {"_fn":"tmp.png", "_cols":800, "_rows":600, "_scols":8, "_srows":8, "_nneighbor":4}
    temp_inf = TmpInf(**template_information)
    return temp_inf, imgQ, imgT, detector, matcher

def main_1(expt_name, fn1, fn2, feature='sift', **template_information):
    import os
    import sys
    imgQ = cv2.imread(fn1, 0)
    imgT = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature)
    if imgQ is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if imgT is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature)
        sys.exit(1)

    temp_inf = TmpInf(**template_information)

    print('using', feature)
    with Timer('calculate Keypoints Descriptors and splitting....'):
        splt_k, splt_d = affine_detect_into_mesh(detector, temp_inf.get_splitnum(), imgQ, simu_param='asift')

    mesh_k_num = np.array([len(keypoints) for keypoints in splt_k]).reshape(temp_inf.get_mesh_shape())

    # mean, median, max, min, peak2peak, standard_deviation, variance = analysis_num(mesh_k_num)
    print("plot mesh keypoint heatmap")
    al_vals = analysis_num(mesh_k_num)
    print("平均, 中央値, 最大値, 最小値, 値の範囲, 標準偏差, 分散")
    print("{0:4f}, {1:4f}, {2:4d}, {3:4d}, {4:4d}, {5:4f}, {6:4f}".format(*al_vals))

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 9))
    h = sns.heatmap(mesh_k_num, annot=True, fmt='g', cmap='Blues')
    h.set(xlabel="x")
    h.set(ylabel="y")
    h.set(title="Heatmap of keypoint amounts -" + temp_inf.tmp_img)
    output_dir = myfm.setup_output_directory(expt_name, "plots")
    h_fig = h.get_figure()
    h_fig.savefig(os.path.join(output_dir, 'meshk_num_'+temp_inf.tmp_img))

    g, df = analysis_kp(splt_k, temp_inf)
    g_fig = g.get_figure()
    g_fig.savefig(os.path.join(output_dir, 'kyepoint_KED_map_'+temp_inf.tmp_img))







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
    splt_kpQ, splt_descQ = affine_detect_into_mesh(detector, splt_num, imgQ, simu_param='default')
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kpT, descT = affine_detect(detector, imgT, pool=pool, simu_param='test')
    print('imgQ - %d features, imgT - %d features' % (count_keypoints(splt_kpQ), len(kpT)))

    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)

    Hs = []
    statuses = []
    kp_pairs_long = []
    Hs_stable = []
    kp_pairs_long_stable = []
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        pairs, H, status = calclate_Homography(pQ, pT, pairs)
        Hs.append(H)
        statuses.append(status)
        if np.sum(status)/len(status) >= 0.4:
            Hs_stable.append(H)
        else:
            Hs_stable.append(None)
        for p in pairs:
            kp_pairs_long.append(p)
            if np.sum(status)/len(status) >= 0.4:
                kp_pairs_long_stable.append(p)

    vis = draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    cv2.imshow('view weak meshes', vis)
    cv2.imwrite('qr1_meshes.png', vis)

    visS = draw_matches_for_meshes(imgQ, imgT, Hs=Hs_stable)
    cv2.imshow('view stable meshes', visS)
    cv2.imwrite('qr1_meshes_stable.png', visS)

    viw = explore_match_for_meshes('affine find_obj', imgQ, imgT, kp_pairs_long_stable, Hs=Hs_stable)
    cv2.imwrite('qr1_mesh_line.png', viw)
    cv2.waitKey()
    cv2.destroyAllWindows()

