#!/usr/bin/env python

'''
SPLIT_ASIFRT　のテスト{}
メッシュ領域毎のメッシュ検出性能を測定
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
from commons.my_common import Timer
import commons.template_info as TmpInf
from make_database import split_affinesim_combinable as slac

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


def make_logger():
    # logging.basicConfig(filename=os.path.join(expt_path, 'log.txt'), level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(lineno)d:%(levelname)5s\n  |>%(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level=logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    timeRotationHandler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(expt_path, fn1 + '_log.txt'),
        when='h',
        interval=24,
        backupCount=3,
        encoding='utf8'
    )
    timeRotationHandler.setLevel(level=logging.INFO)
    timeRotationHandler.setFormatter(formatter)
    logging.getLogger(__name__).addHandler(timeRotationHandler)
    logging.getLogger(__name__).addHandler(consoleHandler)
    logging.getLogger(__name__).setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').addHandler(timeRotationHandler)
    # logging.getLogger('commons').addHandler(consoleHandler)
    logging.getLogger('commons.affine_base').setLevel(level=logging.WARNING)
    logging.getLogger('commons.affine_base').addHandler(timeRotationHandler)
    logging.getLogger('commons.affine_base').addHandler(consoleHandler)
    logging.getLogger('commons.custom_find_obj').setLevel(level=logging.WARNING)
    logging.getLogger('commons.custom_find_obj').addHandler(timeRotationHandler)
    logging.getLogger('commons.custom_find_obj').addHandler(consoleHandler)
    logging.getLogger('commons.my_common').setLevel(level=logging.WARNING)
    logging.getLogger('commons.my_common').addHandler(timeRotationHandler)
    logging.getLogger('commons.my_common').addHandler(consoleHandler)
    # logging.getLogger('make_database').setLevel(level=logging.DEBUG)
    # logging.getLogger('make_database').addHandler(timeRotationHandler)
    # logging.getLogger('make_database').addHandler(consoleHandler)
    logging.getLogger('make_database.split_affinesim').setLevel(level=logging.DEBUG)
    logging.getLogger('make_database.split_affinesim').addHandler(timeRotationHandler)
    logging.getLogger('make_database.split_affinesim').addHandler(consoleHandler)
    logging.getLogger('expt_modules').setLevel(level=logging.DEBUG)
    logging.getLogger('expt_modules').addHandler(timeRotationHandler)
    logging.getLogger('expt_modules').addHandler(consoleHandler)
    logging.getLogger('my_file_path_manager').setLevel(level=logging.DEBUG)
    logging.getLogger('my_file_path_manager').addHandler(timeRotationHandler)
    logging.getLogger('my_file_path_manager').addHandler(consoleHandler)


def check_not_None(arg):
    if arg is not None:
        return len(arg)
    else:
        return 0
if __name__ == '__main__':
    import sys, getopt, os
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    fn1, prefix = args
    expt_path = myfsys.setup_expt_directory(os.path.basename(__file__))
    make_logger()

    logger.info(__doc__.format(os.path.basename(__file__)))
    template_full_fn = myfsys.get_template_file_full_path_(fn1)
    imgQ = cv2.imread(template_full_fn, 0)
    if imgQ is None:
        print('Failed to load fn1:', template_full_fn)
        sys.exit(1)
    print("Using Query: {}".format(fn1))

    detector, matcher = init_feature(emod.Features.SIFT.name)
    if detector is None:
        print('unknown feature:', emod.Features.SIFT.name)
        sys.exit(1)
    print('Using :', emod.Features.SIFT.name)

    scols = 8
    srows = 8
    w, h = imgQ.shape[:2]
    template_fn, ext = os.path.splitext(fn1)
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": w, "_rows": h, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    temp_inf = TmpInf.TemplateInfo(**template_information)

    try:
        with Timer('Lording pickle'):
            splt_kpQ, splt_descQ = slac.affine_load_into_mesh(template_fn, temp_inf.get_splitnum())
    except ValueError as e:
        print(e)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute /Users/tiwasaki/PycharmProjects/makedb/make_split_combine_featureDB_from_templates.py')
        with Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = slac.affine_detect_into_mesh(detector, temp_inf.get_splitnum(),
                                                           imgQ, simu_param='default')

    m_skQ, m_sdQ, m_k_num, merged_map = slac.combine_mesh_compact(splt_kpQ, splt_descQ, temp_inf)
    list_merged_mesh_id = list(set(np.ravel(merged_map)))

    expt_name = os.path.basename(expt_path)

    keyargs = {'prefix_shape': prefix, 'template_fn': template_fn}
    testset_full_path = myfsys.get_dir_full_path_testset('cgs', **keyargs)
    testset_name = os.path.basename(testset_full_path)
    # logger.debug('testset_name is {}'.format(testset_name))
    # logger.info('Test Set:{}'.format(testset_name))
    testcase_fns = os.listdir(testset_full_path)
    testcase_fns.sort()
    logger.debug(testcase_fns)


    output_dir = myfsys.setup_output_directory(expt_name, testset_name, 'outputs', prefix + template_fn)
    detected_dir = myfsys.setup_output_directory(output_dir, 'detected_mesh')
    line_dir = myfsys.setup_output_directory(output_dir, 'dmesh_line')

    vals_list = []
    for fn2 in testcase_fns:
        fn, ext = os.path.splitext(fn2)
        testcase_full_path = os.path.join(testset_full_path, fn2)
        imgT = cv2.imread(testcase_full_path, 0)
        if imgT is None:
            logger.info('Failed to load fn2:', testcase_full_path)
            logger.info('====CONTINUE====')
            continue
        logger.info("Using Training: {}".format(fn2))

        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        with Timer('Detection'):
            kpT, descT = slac.affine_detect(detector, imgT, pool=pool, simu_param='test')
        logger.info('imgQ - %d features, imgT - %d features' % (slac.count_keypoints(splt_kpQ), len(kpT)))

        with Timer('matching'):
            mesh_pQ, mesh_pT, mesh_pairs = slac.match_with_cross(matcher, m_sdQ, m_skQ, descT, kpT)

        # Hs, statuses, pairs = calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs)
        # with Timer('estimation'):
        #     Hs, statuses, pairs = slac.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs)
        Hs = []
        statuses = []
        pairs = []
        def f(*pQpTp):
            inlier_pairs, H, status = calclate_Homography(pQpTp[0], pQpTp[1], pQpTp[2])
            Hs.append(H)
            pairs.extend(inlier_pairs)
            if status is None:
                status = []
            statuses.extend(status)
            return [len(inlier_pairs), len(status), len(pQpTp[2])]

        with Timer('estimation'):
            vals_list = np.array(list(map(f, mesh_pQ, mesh_pT, mesh_pairs)))

        vis = slac.draw_matches_for_meshes(imgQ, imgT, temp_inf=temp_inf, Hs=Hs,
                                           list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)

        cv2.imwrite(os.path.join(detected_dir, fn2 + '.png'), vis)

        viw = slac.explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs,
                                       temp_inf=temp_inf, Hs=Hs, status=statuses,
                                       list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)

        cv2.imwrite(os.path.join(line_dir, fn2 + '.png'), viw)
        cv2.destroyAllWindows()


    keywords = list(map(lambda z: os.path.splitext(z)[0], testcase_fns))
    dictionary = dict(zip(keywords, vals_list))
    np.savez_compressed(os.path.join(output_dir, testset_name), **dictionary)

