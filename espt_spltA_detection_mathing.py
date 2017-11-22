"""
Experimentation for split-ASIFT of
"""
import os
import cv2
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

def read_images(fnQ, fnT):
    import sys
    imgQ = cv2.imread(fnQ, 0)
    imgT = cv2.imread(fnT, 0)
    if imgQ is None:
        print('Failed to load fn1:', fnQ)
        sys.exit(1)

    if imgT is None:
        print('Failed to load fn2:', fnT)
        sys.exit(1)
    return imgQ, imgT

def setup(set_fn):
    """実験開始用ファイル読み込み"""
    # create file handler which logs even debug messages
    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    print(__doc__)

def detect_and_match(detector, matcher, set_fn, splt_num=64, simu_type="default"):
    """SplitA実験"""
    fnQ, t_setn, fnT = set_fn
    imgQ, imgT = read_images(fnQ, fnT)


    print(__doc__)