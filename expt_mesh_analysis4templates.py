#!/usr/bin/env python

'''
SPLIT_ASIFRT　のテスト{}
メッシュ領域毎のメッシュ検出性能を測定
'''

# Python 2/3 compatibility
from __future__ import print_function

import logging.handlers
# built-in modules
from multiprocessing.pool import ThreadPool

#プロット系
import os
if os.getenv('DISPLAY') is None:
    #もし，SSHサーバーサイドで実行するなら，
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
else:
    import matplotlib
    import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import numpy as np

from commons import expt_modules as emod, my_file_path_manager as myfsys
from commons.custom_find_obj import calclate_Homography
from commons.custom_find_obj import init_feature
from commons.my_common import Timer
from make_database import split_affinesim_combinable as slac

# local modules

logger = logging.getLogger(__name__)
def test_module():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    fn1 = os.path.abspath(os.path.join(dir, 'data/templates/qrmarker.png'))
    fn2 = os.path.abspath(os.path.join(dir, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))
    return dir, fn1, fn2

def setup(**kwargs):
    import sys
    imgQ = cv2.imread(kwargs['fn1'], 0)
    imgT = cv2.imread(kwargs['fn2'], 0)
    detector, matcher = init_feature(kwargs['feature'])
    if imgQ is None:
        print('Failed to load fn1:', kwargs['fn1'])
        sys.exit(1)

    if imgT is None:
        print('Failed to load fn2:', kwargs['fn2'])
        sys.exit(1)

    if detector is None:
        print('unknown feature:', kwargs['feature'])
        sys.exit(1)

    temp_inf = slac.TmpInf(**kwargs['template_information'])
    return imgQ, imgT, detector, matcher, temp_inf


def main_1(expt_name, fn1, fn2, feature='sift', **template_information):
    import sys
    kw= {'fn1':fn1, 'fn2':fn2, 'feature':feature, 'template_information':template_information}
    # imgQ = cv2.imread(fn1, 0)
    # imgT = cv2.imread(fn2, 0)
    # detector, matcher = init_feature(feature)
    # if imgQ is None:
    #     print('Failed to load fn1:', fn1)
    #     sys.exit(1)
    #
    # if imgT is None:
    #     print('Failed to load fn2:', fn2)
    #     sys.exit(1)
    #
    # if detector is None:
    #     print('unknown feature:', feature)
    #     sys.exit(1)
    #
    # temp_inf = slac.TmpInf(**template_information)
    imgQ, imgT, detector, matcher, temp_inf = setup(**kw)
    pass


    print('using', feature)
    with Timer('calculate Keypoints Descriptors and splitting....'):
        splt_k, splt_d = slac.affine_detect_into_mesh(detector, temp_inf.get_splitnum(), imgQ, simu_param='asift')

    mesh_k_num = np.array([len(keypoints) for keypoints in splt_k]).reshape(temp_inf.get_mesh_shape())

    # mean, median, max, min, peak2peak, standard_deviation, variance = analysis_num(mesh_k_num)
    print("plot mesh keypoint heatmap")
    al_vals = slac.analysis_num(mesh_k_num)
    print("平均, 中央値, 最大値, 最小値, 値の範囲, 標準偏差, 分散")
    print("{0:4f}, {1:4f}, {2:4d}, {3:4d}, {4:4d}, {5:4f}, {6:4f}".format(*al_vals))


    sns.set_context("poster")
    # plt.figure(figsize=(12, 9))
    h = sns.heatmap(mesh_k_num, annot=True, fmt='g', cmap='Blues')
    h.set(xlabel="x")
    h.set(ylabel="y")
    h.set(title="Heatmap of keypoint amounts -" + temp_inf.tmp_img)
    output_dir = slac.myfm.setup_output_directory(expt_name, "plots")
    h_fig = h.get_figure()
    h_fig.savefig(os.path.join(output_dir, 'meshk_num_'+temp_inf.tmp_img))

    df = slac.analysis_kp(splt_k, temp_inf)

    # with Timer('plotting Kernel De'):
    #     for i in range(temp_inf.get_splitnum()):
    #         ax = sns.kdeplot(df.query('mesh_id == ' + str(i))['x'], df.query('mesh_id == ' + str(i))['y'], shade=True)
    #         ax.set(ylim=(600, 0))
    #         ax.set(xlim=(0, 800))
    #         ax.set(xlabel="x")
    #         ax.set(ylabel="y")
    #         ax.set(title="Kernel density estimation")

    g = sns.kdeplot(df['x'], df['y'], shade=True, )
    g.set(ylim=(600, 0))
    g.set(xlim=(0, 800))
    g.set(xlabel="Width of image")
    g.set(ylabel="Height of image")
    g.set(title="Kernel density estimation-"+temp_inf.tmp_img)
    g_fig = g.get_figure()
    g_fig.savefig(os.path.join(output_dir, 'Kyepoint_KED_map_'+temp_inf.tmp_img))


if __name__ == '__main__':
    test_module = test_module()
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

    logger.info(__doc__.format(os.path.basename(__file__)))
    a = myfsys.make_list_template_filename()
    a = emod.only(a, 'glass.png')
    expt_name = os.path.basename(expt_path)
    for template_fn in a:
        template_information = {"_fn":"tmp.png", "template_img":template_fn,
                                "_cols":800, "_rows":600, "_scols":8, "_srows":8, "_nneighbor":4}
        logger.info('Template:{}'.format(template_fn))
        # global s_kpQ, s_descQ, testset_full_path
        template_full_fn = myfsys.get_template_file_full_path_(template_fn)
        # imgQ, s_kpQ, s_descQ = split_asift_detect(detector, template_full_fn, split_num=split_num)

        # keyargs = {'prefix_shape': emod.PrefixShapes.PL.value, 'template_fn': template_fn}
        # testset_full_path = myfsys.get_dir_full_path_testset('cgs', **keyargs)
        # testset_name = os.path.basename(testset_full_path)
        # logger.debug('testset_name is {}'.format(testset_name))
        # logger.info('Test Set:{}'.format(testset_name))
        # output_dir = myfsys.setup_output_directory(expt_name, testset_name, 'npfiles')
        # testcase_fns = os.listdir(testset_full_path)
        # testcase_fns.sort()
        # print(testcase_fns)

        main_1(expt_name, fn1=template_full_fn, fn2=test_module[2], **template_information)

