import enum
import os
import my_file_system as myfsys
from my_file_system import DirNames

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


def setup_expt_directory(base_name):
    outputs_dir = myfsys.get_dir_path_(DirNames.OUTPUTS.value)
    expt_name, ext = os.path.splitext(base_name)
    expt_path = os.path.join(outputs_dir, expt_name)
    if os.path.exists(expt_path):
        return expt_path
    os.mkdir(expt_path)
    return expt_path
