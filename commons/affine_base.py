# Python 2/3 compatibility
from __future__ import print_function

# built-in modules

import cv2
import numpy as np
from logging import getLogger

# local modules
from commons.custom_find_obj import filter_matches_wcross as c_filter


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai

def a_detect(p, detector, img):
    """Affine Skeyして特徴点計算をする"""
    t, phi = p
    timg, tmask, Ai = affine_skew(t, phi, img)
    keypoints, descrs = detector.detectAndCompute(timg, tmask)
    for kp in keypoints:
        x, y = kp.pt
        kp.pt = tuple(np.dot(Ai, (x, y, 1)))
    if descrs is None:
        descrs = []
    return keypoints, descrs

def w_a_detect(args):
    a_detect(*args)

def calc_affine_params(simu: str ='default') -> list:
    """
    Calculation affine simulation parameter tilt and phi
    You get list object of sets (tilt, phi) as taple
    :param simu: set simulation taype
    :return: list of taple
    """
    params = [(1.0, 0.0)]
    if simu == 'default' or simu == 'asift' or simu is None:
        simu = 'default'
        for t in 2**(0.5*np.arange(1, 6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))

    if simu == 'degrees':
        """半周する"""
        for t in np.reciprocal(np.cos(np.radians(np.arange(10, 90, 10)))):
            for phi in np.arange(0, 180, 20):
                params.append((t, phi))

    if simu == 'degrees-full':
        """一周する"""
        for t in np.reciprocal(np.cos(np.radians(np.arange(10, 90, 10)))):
            for phi in np.arange(0, 360, 20):
                params.append((t, phi))

    if simu == 'test2':
        print("This simulation is Test2 type")
        for t in np.reciprocal(np.cos(np.radians(np.arange(10, 11, 10)))):
            for phi in np.arange(0, 21, 20):
                params.append((t, phi))

    if simu == 'test' or simu == 'sift':
        print("This simulation is Test type")
        pass

    print("%s -type params: %d" % (simu, len(params)))
    return params


def affine_detect(detector, img, mask=None, pool=None, simu_param=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = calc_affine_params(simu_param)
    if len(params) == 1:
        keypoints, descrs = detector.detectAndCompute(img, mask)
        return keypoints, np.array(descrs)

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img, mask)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = list(map(f, params))
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

def match_with_cross(matcher, descQ, kpQ, descT, kpT):
    raw_matchesQT = matcher.knnMatch(descQ, trainDescriptors=descT, k=2)
    raw_matchesTQ = matcher.knnMatch(descT, trainDescriptors=descQ, k=2)
    pQ, pT, pairs = c_filter(kpQ, kpT, raw_matchesQT, raw_matchesTQ)
    return pQ, pT, pairs
