import enum
import cv2
import logging
import inspect
import numpy as np
import my_file_path_manager as myfsys
from my_file_path_manager import DirNames

logger = logging.getLogger(__name__)

class Features(enum.Enum):
    SIFT = enum.auto()
    SURF = enum.auto()
    AKAZE = enum.auto()
    ORB = enum.auto()
    BRISK = enum.auto()
    SIFT_FLANN = enum.auto()
    SURF_FLANN = enum.auto()
    AKAZE_FLANN = enum.auto()
    ORB_FLANN = enum.auto()
    BRISK_FLANN = enum.auto()

class PrefixShapes(enum.Enum):
    MLTF = 'mltf_'
    PL = 'pl_'
    CRV = 'crv_'
    ALL = 'all_'

def only(obj_list, only_obj):
    """実験対象を絞りたい時"""
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    return [obj_list[obj_list.index(only_obj)]]

def read_image(fn):
    import sys
    img = cv2.imread(fn, 0)
    if img is None:
        logger.error('Failed to load fn1:{0}'.format(fn))
        sys.exit(1)
    return img

def detect(detector, fn, mask=None):
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    img = read_image(fn)
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    keypoints, descrs = detector.detectAndCompute(img, mask)
    return img, keypoints, descrs

