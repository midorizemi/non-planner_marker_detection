from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce
# built-in modules
from contextlib import contextmanager
from commons.common import clock
from logging import getLogger

logger = getLogger(__name__)

@contextmanager
def Timer(msg):
    logger.info('Measuring Time {}'.format(msg))
    start = clock()
    try:
        yield
    finally:
        logger.info("%.2f ms\n" % ((clock()-start)*1000))
