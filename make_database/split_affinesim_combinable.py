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

# 解析
import pandas as pd

import cv2
import numpy as np
import os

# local modules
from commons.common import Timer
from commons.common import anorm
from commons.my_common import set_trace, debug, load_pikle
from commons.find_obj import init_feature
from commons.affine_base import affine_detect
from commons.template_info import TemplateInfo as TmpInf
from commons.custom_find_obj import filter_matches_wcross as c_filter
from commons.custom_find_obj import calclate_Homography4splitmesh
from commons import my_file_path_manager as myfm


def split_kd(keypoints, descrs, splt_num):
    tmp = TmpInf()
    split_tmp_img = tmp.make_splitmap()
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
        if len(splits_k[i]) == len(splits_d):
            sys.exit(1)
        splits_d[i] = np.array(split_d, dtype=np.float32)

    return splits_k, splits_d


def combine_mesh(splt_k, splt_d, temp_inf):
    """
    :type temp_inf: TmpInf
    :param splt_k:
    :param splt_d:
    :param temp_inf:
    :return:
    """
    mesh_map = temp_inf.get_mesh_map()
    mesh_k_num = np.array([len(keypoints) for keypoints in splt_k]).reshape(temp_inf.get_mesh_shape())

    for i, kd in enumerate(zip(splt_k, splt_d)):
        """
        矩形メッシュをマージする．4近傍のマージ．順番は左，上，右，下．
        最大値のところとマージする
        :return:
        """
        meshid_list = temp_inf.get_meshidlist_nneighbor(i)
        self_id = mesh_map[temp_inf.get_meshid_index(i)]
        self_k_num = mesh_k_num[temp_inf.get_meshid_index(self_id)]
        if not self_id == i or len(np.where(mesh_map == i)[0]) > 1 or self_k_num == 0:
            """すでにマージされている"""
            continue
        # 最大値のindexを求める．
        dtype = [('muki', int), ('keypoint_num', int), ('merge_id', int)]
        # tmp = np.array([list(len(meshid_list)-index, mesh_k_num[temp_inf.get_meshid_vertex(id)], id )
        #                for index, id in enumerate(meshid_list) if id is not None]).astype(np.int64)
        tmp = []
        # for index, id in enumerate(meshid_list):
        #     if id is not None:
        #         tmp.extend([len(meshid_list) - index, mesh_k_num[temp_inf.get_meshid_vertex(id)], id])
        # tmp = np.array(tmp)reshape(int(len(tmp)/3, 3).astype(np.int64)
        try:
            for index, id in enumerate(meshid_list):
                if id is not None:
                    tmp.append([len(meshid_list) - index, mesh_k_num[temp_inf.get_meshid_index(id)], id])
        except(IndexError):
            set_trace()

        tmp = np.array(tmp).astype(np.int64)
        median_nearest = np.median(tmp[:, 1])
        if median_nearest < self_k_num:
            # TODO マージ判定
            # 近傍中の中央値よりも注目メッシュのキーポイント数が大きい場合は無視する
            continue
        tmp.dtype = dtype
        tmp.sort(order=['keypoint_num', 'muki'])  # 左回りでかつキーポイント数が最大
        # idにself_idをマージする, 昇順なので末端
        merge_id = tmp[-1][0][2]
        mesh_map[temp_inf.get_meshid_index(i)] = merge_id
        splt_k[merge_id].extend(kd[0])
        mesh_k_num[temp_inf.get_meshid_index(merge_id)] = mesh_k_num[temp_inf.get_meshid_index(merge_id)] + self_k_num
        try:
            np.concatenate((splt_d[merge_id], kd[1]))
        except(IndexError, ValueError):
            set_trace()

        # マージされて要らなくなったメッシュは消す
        splt_k[i] = None
        splt_d[i] = None
        mesh_k_num[temp_inf.get_meshid_index(self_id)] = 0

    return splt_k, splt_d, mesh_k_num, mesh_map


def combine_mesh_compact(splt_k, splt_d, temp_inf):
    sk, sd, mesh_k_num, merged_map = combine_mesh(splt_k, splt_d, temp_inf)
    m_sk = compact_merged_splt(sk)
    m_sd = compact_merged_splt(sd)
    return m_sk, m_sd, mesh_k_num, merged_map


def compact_merged_splt(m_s):
    return [x for x in m_s if x is not None]


def affine_detect_into_mesh(detector, split_num, img1, mask=None, simu_param='default'):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp, desc = affine_detect(detector, img1, mask, pool=pool, simu_param=simu_param)
    return split_kd(kp, desc, split_num)


def affine_load_into_mesh(template_fn, splt_num):
    pikle_path = myfm.get_pikle_path(template_fn)
    if not os.path.exists(pikle_path):
        print('Not found {}'.format(pikle_path))
        raise ValueError('Failed to load pikle:', pikle_path)
    kp, des = load_pikle(pikle_path)
    return split_kd(kp, des, splt_num)


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
        if kps is not None:
            len_s_kp += len(kps)
    return len_s_kp


def merge_rule(id, mean, mesh_k_num, temp_inf):
    """
    #TODO
    何かしらのマージルール
    特徴点数とか，分布とか，特徴量とかでマージが必要なメッシュかをかをはんていする
    :type mesh_k_num :np.ndarray
    """

    if mean > mesh_k_num[temp_inf.get_meshid_index(id)]:
        pass


def analysis_num(mesh_k_num):
    # 分析１：特徴点数のバラつき
    mean = mesh_k_num.mean()
    median = np.median(mesh_k_num)
    max = np.amax(mesh_k_num)
    min = np.amin(mesh_k_num)
    peak2peak = np.ptp(mesh_k_num)
    standard_deviation = np.std(mesh_k_num)
    variance = np.var(mesh_k_num)
    return mean, median, max, min, peak2peak, standard_deviation, variance


def analysis_kp(splt_k, temp_inf: TmpInf) -> pd.DataFrame:
    """
    #分析2：特徴点座標のバラつき
    :param splt_k:
    :param temp_inf:
    :return:
    """
    marker_k_np = [[np.int32(i), np.int32(kp.pt[0]), np.int32(kp.pt[1])] for i, keypoints in enumerate(splt_k)
                   for kp in keypoints]
    df = pd.DataFrame(marker_k_np, columns=['mesh_id', 'x', 'y'])
    print("Done make data")
    print(df.head(5))
    return df


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

    template_information = {"_fn": "tmp.png", "_cols": 800, "_rows": 600, "_scols": 8, "_srows": 8, "_nneighbor": 4}
    temp_inf = TmpInf(**template_information)
    return temp_inf, imgQ, imgT, detector, matcher

def get_id_list(_id, tmp_inf, merged_mesh_map: np.ndarray):
    flag = np.where(merged_mesh_map == _id, True, False)
    map = tmp_inf.get_mesh_map()
    return map[flag].tolist()

def explore_meshes(imgT, temp_inf, Hs=None, list_merged_mesh_id=None, mesh_map=None):
    hT, wT = imgT.shape[:2]

    meshes = []
    for H, mid in zip(Hs, list_merged_mesh_id):
        list_ms = get_id_list(mid, temp_inf, mesh_map)
        rectangles_vertexes = temp_inf.get_mesh_recanglarvertex_list(list_ms)

        for vertexes in rectangles_vertexes:
            if H is not None:
                corners = np.int32(cv2.perspectiveTransform(vertexes.reshape(1, -1, 2), H).reshape(-1, 2) + (wT, 0))
                meshes.append(corners)

    return meshes


def draw_matches_for_meshes(imgT, imgQ, temp_inf, Hs=None, vis=None, list_merged_mesh_id=None, merged_map=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    if vis is None:
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = imgT
        vis[:h2, w1:w1 + w2] = imgQ
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    meshes = explore_meshes(imgT,temp_inf, Hs, list_merged_mesh_id, merged_map)
    for mesh_corners in meshes:
        cv2.polylines(vis, [mesh_corners], True, (25, 94, 255), thickness=3, lineType=cv2.LINE_AA)

    return vis

def explore_match_for_meshes(win, imgT, imgQ, kp_pairs, temp_inf=None, status=None, Hs=None,
                             list_merged_mesh_id=None, merged_map=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = imgT
    vis[:h2, w1:w1 + w2] = imgQ
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    vis = draw_matches_for_meshes(imgT, imgQ, temp_inf,  Hs, vis,
                                  list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
    vis0 = vis.copy()
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    # cv2.imshow(win, vis)

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i]]
                cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:, w1:] = cv2.drawKeypoints(cur_vis[:, w1:], kp2s, None, flags=4, color=kp_color)

        # cv2.imshow(win, cur_vis)

    # cv2.setMouseCallback(win, onmouse)
    return vis

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
        fn1 = os.path.abspath(os.path.join(dir, 'data/templates/menko.png'))
        # fn2 = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))
        # fn2 = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/unittest/011_080-100.png'))
        fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/unittest/219_020-020.png'))

    imgQ = cv2.imread(fn1, 0)
    imgT = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature_name)

    if imgQ is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)
    print("Using Query: {}".format(fn1))
    if imgT is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)
    print("Using Training: {}".format(fn2))
    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    print('Using :', feature_name)

    template_fn, ext = os.path.splitext(os.path.basename(fn1))
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": 800, "_rows": 600, "_scols": 8, "_srows": 8, "_nneighbor": 4}
    temp_inf = TmpInf(**template_information)
    try:
        with Timer('Lording pickle'):
            splt_kpQ, splt_descQ = affine_load_into_mesh(template_fn, temp_inf.get_splitnum())
    except ValueError as e:
        print(e)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute /Users/tiwasaki/PycharmProjects/makedb/make_split_combine_featureDB_from_templates.py')
        with Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = affine_detect_into_mesh(detector, temp_inf.get_splitnum(),
                                                           imgQ, simu_param='default')

    sk_num = count_keypoints(splt_kpQ)
    m_skQ, m_sdQ, m_k_num, merged_map = combine_mesh_compact(splt_kpQ, splt_descQ, temp_inf)
    if not sk_num == count_keypoints(m_skQ) and not count_keypoints(m_skQ) == np.sum(m_k_num):
        print('{0}, {1}, {2}'.format(sk_num, count_keypoints(m_skQ), np.sum(m_k_num)))
        sys.exit(1)
    median = np.nanmedian(m_k_num)
    list_merged_mesh_id = list(set(np.ravel(merged_map)))

    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection'):
        kpT, descT = affine_detect(detector, imgT, pool=pool, simu_param='test')
    print('imgQ - %d features, imgT - %d features' % (count_keypoints(splt_kpQ), len(kpT)))

    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = match_with_cross(matcher, m_sdQ, m_skQ, descT, kpT)

    # Hs, statuses, pairs = calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs)
    with Timer('estimation'):
        Hs, statuses, pairs = calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)

    vis = draw_matches_for_meshes(imgQ, imgT, temp_inf=temp_inf, Hs=Hs, list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
    cv2.imshow('view weak meshes', vis)
    cv2.imwrite('qrmarker_detection_merged.png', vis)
    cv2.waitKey()

    # viw = explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs,
    #                                temp_inf=temp_inf, Hs=Hs,
    #                                list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
    #
    # cv2.imwrite('qr1_mesh_line.png', viw)
    cv2.waitKey()
    cv2.destroyAllWindows()
