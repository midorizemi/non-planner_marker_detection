#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

from commons.template_info import TemplateInfo as TmpInf
from commons.common import anorm


def filter_matches_wcross(kp_T, kp_Q, matchesTQ, matchesQT, ratio=0.75):
    """
    filtering matches with cross check
    :param kp_T: Train keypoints
    :param kp_Q: Query keypoints
    :param matchesTQ: Train -> Query dmach list
    :param matchesQT: Query -> Train
    :param ratio: ratio check parameter
    :return: filterd matched points and pairs
    """

    def ratiotest(matches):
        """Dont Use"""
        dmatches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                dmatches.append(m)
        return dmatches

    def ratiotest_cross(matches):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp_T[m.trainIdx])
                mkp2.append(kp_Q[m.queryIdx])
        return mkp1, mkp2

    def crosscheck(dmatchesTQ, dmatchesQT):
        """Dont Use"""
        mkp1, mkp2 = [], []
        for forward in dmatchesTQ:
            if len(dmatchesQT) >= forward.trainIdx:
                print('TQ.length = %d, QT.length = %d' % (len(dmatchesTQ), len(dmatchesQT)))
                print('TQ.t=%d, TQ.q=%d' % (forward.trainIdx, forward.queryIdx))
            backward = dmatchesQT[forward.trainIdx]
            if backward.trainIdx == forward.queryIdx:
                mkp1.append(kp_T[forward.trainIdx])
                mkp2.append(kp_Q[forward.queryIdx])
        return mkp1, mkp2

    def crossCheck(dmatchesTQ, dmatchesQT):
        crossOK = []
        for forward in dmatchesTQ:
            if len(dmatchesQT) <= forward[0].trainIdx:
                print('TQ.length = %d, QT.length = %d' % (len(dmatchesTQ), len(dmatchesQT)))
                print('TQ.t=%d, TQ.q=%d' % (forward[0].trainIdx, forward[0].queryIdx))
            backward = dmatchesQT[forward[0].trainIdx]
            if backward[0].trainIdx == forward[0].queryIdx:
                crossOK.append(forward)
        return crossOK

    # dmatches12 = ratiotest(matchesTQ)
    # dmatches21 = ratiotest(matchesQT)
    # mkp1, mkp2 = crosscheck(dmatches12, dmatches21)
    dmatches_cross = crossCheck(matchesTQ, matchesQT)
    mkp1, mkp2 = ratiotest_cross(dmatches_cross)
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def explore_meshes(imgT, Hs=None):
    hT, wT = imgT.shape[:2]
    tmp = TmpInf()

    meshes = []
    for id, H in enumerate(Hs):
        mesh_corners = tmp.calculate_mesh_corners(id)
        if H is not None:
            corners = np.int32(cv2.perspectiveTransform(mesh_corners.reshape(1, -1, 2), H).reshape(-1, 2) + (wT, 0))
            meshes.append(corners)

    return meshes


def draw_matches_for_meshes(imgT, imgQ, kp_pairs, status=None, Hs=None, vis=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    if vis is None:
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = imgT
        vis[:h2, w1:w1 + w2] = imgQ
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    meshes = explore_meshes(imgT, Hs)
    for mesh_corners in meshes:
        cv2.polylines(vis, [mesh_corners], True, (255, 0, 0))

    return vis


def explore_match_for_meshes(win, imgT, imgQ, kp_pairs, status=None, Hs=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = imgT
    vis[:h2, w1:w1 + w2] = imgQ
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis0 = vis.copy()

    vis = draw_matches_for_meshes(imgT, imgQ, kp_pairs, status, Hs, vis)
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

    cv2.imshow(win, vis)

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

        cv2.imshow(win, cur_vis)

    cv2.setMouseCallback(win, onmouse)
    return vis


def calclate_Homography(pT, pQ, pairs):
    if len(pQ) >= 4:
        H, status = cv2.findHomography(pT, pQ, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        # do not draw outliers (there will be a lot of them)
        pairs = [kpp for kpp, flag in zip(pairs, status) if flag]
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(pQ))

    return pairs, H, status
    pass
