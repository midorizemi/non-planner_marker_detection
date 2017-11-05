# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from commons.template_info import TemplateInfo as tmpinf

def make_splitmap(split_tmp):
    assert isinstance(split_tmp, tmpinf)
    img = np.zeros((split_tmp.rows, split_tmp.cols, 1), np.uint8)
    for i in range(split_tmp.get_splitnum()):
        x, y = split_tmp.colculate_mesh_tlvertex(i)
        mesh = np.tile(np.uint8([i]), (split_tmp.offset_r, split_tmp.offset_c, 1))
        img[y:y+split_tmp.offset_r, x:x+split_tmp.offset_c] = mesh

    return img


if __name__ == '__main__':
    tmp = tmpinf("inputs/templates/mesh_labelpy.png")
    tmp_img = make_splitmap(tmp)
    cv2.imshow("test", tmp_img)
    cv2.imwrite(tmp.fn, tmp_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

