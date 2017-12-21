#!/usr/bin/env python

'''
SPLIT_ASIFRT　のテスト
メッシュ領域毎のメッシュ検出性能を測定
'''

# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import os
import logging


# local modules
from nltk.chat.rude import pairs

from commons.my_common import Timer
from commons.find_obj import filter_matches, explore_match
from commons.custom_find_obj import init_feature
from make_database import split_affinesim as saf
import my_file_path_manager as myfsys
import expt_modules as emod
from commons.custom_find_obj import calclate_Homography

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


if __name__ == '__main__':
    expt_path = myfsys.setup_expt_directory(os.path.basename(__file__))
    logging.basicConfig(level=logging.DEBUG)
    a = myfsys.make_list_template_filename()
    # a = emod.only(a, 'qrmarker.png')
    detector, matcher = init_feature(emod.Features.SIFT.name)
    column_num = 8
    row_num = 8
    split_num = column_num * row_num
    expt_name = os.path.basename(expt_path)
    for template_fn in a:
        template_full_fn = myfsys.get_template_file_full_path_(template_fn)
        imgQ, s_kpQ, s_descQ = split_asift_detect(detector, template_full_fn, split_num=split_num)

        keyargs = {'prefix_shape': emod.PrefixShapes.PL.value, 'template_fn': template_fn}
        testset_full_path = myfsys.get_dir_full_path_testset('cgs', **keyargs)
        testset_name = os.path.basename(testset_full_path)
        testcase_fns = os.listdir(testset_full_path)
        testcase_fns.sort()
        output_dir = myfsys.setup_output_directory(expt_name, testset_name[0], 'npfiles')
        # testcase_fns = emod.only(testcase_fns, '288_010-350.png')
        def clc(testcase_fn):
            testcase_full_path = os.path.join(testset_full_path, testcase_fn)
            imgT, kpT, descT = emod.detect(detector, testcase_full_path)
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
        dictionary = dict(zip(keywords, results))
        np.savez_compressed(os.path.join(expt_path, testset_name), **dictionary)

