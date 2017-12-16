import enum

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
