#!/usr/bin/env python

'''
{} > ARマーカを検出します
'''

# Python 2/3 compatibility
from __future__ import print_function

import logging.handlers
import os
# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from commons import expt_modules as emod, my_file_path_manager as myfsys
from commons.custom_find_obj import calclate_Homography
from commons.custom_find_obj import init_feature
from commons.custom_find_obj import draw_matches_for_meshes
from commons.my_common import Timer
from make_database import split_affinesim as saf

# local modules

logger = logging.getLogger(__name__)

def split_asift_detect(detector, fn, split_num):
    img = emod.read_image(fn)
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection with [ ASIFT ]'):
        splt_kp, splt_desc = saf.affine_detect_into_mesh(detector, split_num, img, simu_param='asift')
    return img, splt_kp, splt_desc

def expt_meshdtct_perf_spltASIFT(column_num = 8, row_num = 8, template_f = 'qrmarker.png'):
    split_num = column_num * row_num
    template_fn = myfsys.get_template_file_full_path_(template_f)
    detector, matcher = init_feature(emod.Features.SIFT)
    imgQ, s_kpQ, s_descQ = split_asift_detect(detector, template_fn, split_num=split_num)
    keyargs = {'prefix_shape': emod.PrefixShapes.PL.value, 'template_fn': template_fn}

    testset_full_path = myfsys.get_dir_full_path_testset('cgs', **keyargs)
    print(testset_full_path)
    testcase_fns = os.listdir(testset_full_path)
    testcase_fns.sort()
    print(testcase_fns)
    results = []
    for input_fns in testcase_fns:
        calculate_each_mesh(column_num, detector, input_fns, matcher, results, row_num, s_descQ, s_kpQ)

def calculate_each_mesh(column_num, detector, input_fns, matcher, results, row_num, s_descQ, s_kpQ):
    imgT, kpT, descT = emod.detect(detector, input_fns)
    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = saf.match_with_cross(matcher, s_descQ, s_kpQ, descT, kpT)

    def f(pQ, pT, p):
        inlier_pairs, H, status = calclate_Homography(pQ, pT, p)
        if status is None:
            status = []
        return [len(inlier_pairs), len(status), len(p)]

    pairs_on_meshes = np.array(list(map(f, zip(mesh_pQ, mesh_pT, mesh_pairs))))
    # pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    # pairs_on_mesh_list = np.array(pool.imap(f, zip(range(len(mesh_pQ)), mesh_pQ, mesh_pT, mesh_pairs)))
    # pairs_on_mesh = pairs_on_mesh.reshape(row_num, column_num)
    results.append(pairs_on_meshes.reshape(row_num, column_num))


def exam(testset_full_path,  s_kpQ, s_descQ ):
    testcase_fns = os.listdir(testset_full_path)
    testcase_fns.sort()

    # testcase_fns = emod.only(testcase_fns, '288_010-350.png')
    def clc(testcase_fn):
        logger.info('Test Case:{}'.format(testcase_fn))
        testcase_full_path = os.path.join(testset_full_path, testcase_fn)
        imgT, kpT, descT = emod.detect(detector, testcase_full_path)
        if len(kpT) == 0:
            return np.zeros((row_num, column_num, 3))
        with Timer('matching'):
            mesh_pQ, mesh_pT, mesh_pairs = saf.match_with_cross(matcher, s_descQ, s_kpQ, descT, kpT)

        def f(*pQpTp):
            inlier_pairs, H, status = calclate_Homography(pQpTp[0], pQpTp[1], pQpTp[2])
            if status is None:
                status = []
            return [len(inlier_pairs), len(status), len(pQpTp[2])]

        pairs_on_meshes = np.array(list(map(f, mesh_pQ, mesh_pT, mesh_pairs)))

        return pairs_on_meshes.reshape(row_num, column_num, 3)

    with Timer('matching'):
        results = list(map(clc, testcase_fns))
    # results = np.array(list(map(clc, testcase_fns)))
    keywords = list(map(lambda z: os.path.splitext(z)[0], testcase_fns))
    return dict(zip(keywords, results))


def calculate_hompgraphy(mesh_pQ, mesh_pT, mesh_pairs):
    list_H = []
    statuses = []
    kp_pairs = []
    pairs_on_meshes = np.empty(0)
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        inlier_pairs, H, status = calclate_Homography(pQ, pT, pairs)
        list_H.append(H)
        statuses.append(status)
        kp_pairs.append(inlier_pairs)
        if status is None:
            status = []
        pairs_on_meshes = np.append(pairs_on_meshes, np.array([len(inlier_pairs), len(status), len(pairs)]))
    return list_H, statuses, kp_pairs, pairs_on_meshes.reshape((row_num, column_num, 3))

if __name__ == '__main__':
    expt_path = myfsys.setup_expt_directory(os.path.basename(__file__))
    # logging.basicConfig(filename=os.path.join(expt_path, 'log.txt'), level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(lineno)d:%(levelname)5s\n  |>%(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level=logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    timeRotationHandler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(expt_path, 'log.txt'),
        when='h',
        interval=24,
        backupCount=3,
        encoding='utf8'
    )
    timeRotationHandler.setLevel(level=logging.DEBUG)
    timeRotationHandler.setFormatter(formatter)
    logging.getLogger(__name__).addHandler(timeRotationHandler)
    logging.getLogger(__name__).addHandler(consoleHandler)
    logging.getLogger(__name__).setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').addHandler(timeRotationHandler)
    # logging.getLogger('commons').addHandler(consoleHandler)
    logging.getLogger('commons.affine_base').addHandler(timeRotationHandler)
    logging.getLogger('commons.affine_base').addHandler(consoleHandler)
    logging.getLogger('commons.custom_find_obj').addHandler(timeRotationHandler)
    logging.getLogger('commons.custom_find_obj').addHandler(consoleHandler)
    logging.getLogger('commons.my_common').addHandler(timeRotationHandler)
    logging.getLogger('commons.my_common').addHandler(consoleHandler)
    # logging.getLogger('make_database').setLevel(level=logging.DEBUG)
    # logging.getLogger('make_database').addHandler(timeRotationHandler)
    # logging.getLogger('make_database').addHandler(consoleHandler)
    logging.getLogger('make_database.split_affinesim').addHandler(timeRotationHandler)
    logging.getLogger('make_database.split_affinesim').addHandler(consoleHandler)
    logging.getLogger('commons.expt_modules').addHandler(timeRotationHandler)
    logging.getLogger('commons.expt_modules').addHandler(consoleHandler)
    logging.getLogger('commons.my_file_path_manager').addHandler(timeRotationHandler)
    logging.getLogger('commons.my_file_path_manager').addHandler(consoleHandler)

    logger.info(__doc__.format(os.path.basename(__file__)))

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2, column_num, row_num = args
    except:
        dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        fn1 = myfsys.get_template_file_full_path_('qrmarker.png')
        fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/cgs/pl_qrmarker/288_010-350.png'))
        column_num = 8
        row_num = 8

    detector, matcher = init_feature(feature_name)

    if detector is None:
        logger.info('unknown feature:{}'.format(feature_name))
        sys.exit(1)

    split_num = column_num * row_num
    img_q, splt_kp_q, splt_desc_q = split_asift_detect(detector, fn1, split_num)

    logger.debug('using {}'.format(feature_name))

    img_t, kp_t, desc_t = emod.detect(detector, fn2)
    print('imgQ - %d features, imgT - %d features' % (saf.count_keypoints(splt_kp_q), len(kp_t)))

    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = saf.match_with_cross(matcher, splt_desc_q, splt_kp_q, desc_t, kp_t)

    list_H, statuses, kp_pairs, pairs_on_meshes = calculate_hompgraphy(mesh_pQ, mesh_pT, mesh_pairs)

    mt_p = pairs_on_meshes[:, :, 0]
    mt_s = pairs_on_meshes[:, :, 1]
    mt_m = pairs_on_meshes[:, :, 2]
    ratio_ps = mt_p / mt_s
    ratio_pm = mt_p / mt_m
    logger.info(mt_p)
    logger.info(mt_s)
    logger.info(mt_m)
    logger.info(ratio_ps)
    logger.info(ratio_pm)

    vis = draw_matches_for_meshes(img_q, img_t, Hs=list_H)

    test_set_name = os.path.dirname(fn2)
    test_case_name, ext = os.path.splitext(os.path.basename(fn2))

    cv2.imwrite(os.path.join(expt_path, test_set_name, test_case_name + '_hconcat_ditection.png'), vis)
    np.savetxt(os.path.join(expt_path, test_set_name, test_case_name + '_pairs_on_meshes.csv'), pairs_on_meshes, delimiter=',')


