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
from make_database import split_affinesim_combinable as slac

# local modules

logger = logging.getLogger(__name__)

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
    # a = emod.only(a, 'glass.png')
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

        slac.main_1(expt_name, fn1=template_full_fn, fn2='test.png', **template_information)

