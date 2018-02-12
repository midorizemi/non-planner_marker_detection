# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from commons.template_info import TemplateInfo as tmpinf

def make_splitmap(split_tmp):
    assert isinstance(split_tmp, tmpinf)
    img = np.zeros((split_tmp.rows, split_tmp.cols, 1), np.uint8)
    for i in range(split_tmp.get_splitnum()):
        y, x = split_tmp.calculate_mesh_topleftvertex(i)
        mesh = np.tile(np.uint8([i]), (split_tmp.offset_r, split_tmp.offset_c, 1))
        img[y:y+split_tmp.offset_r, x:x+split_tmp.offset_c] = mesh

    return img


if __name__ == '__main__':
    scols = 8
    srows = 8
    template_information = {"_fn": "inputs/templates/mesh_1_labelpy.png", "template_img": "qrmarker.png",
                            "_cols": 800, "_rows": 600, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    tmp = tmpinf(**template_information)
    tmp_img = tmp.make_splitmap()
    # tmp_img = make_splitmap(tmp)
    cv2.imshow("test", tmp_img)
    # cv2.imwrite(tmp.fn, tmp_img)
    cv2.waitKey()

    wier = np.tile(np.uint8(255), (600, 800, 3))
    for i in range(tmp.get_splitnum()):
        corners = np.int32(tmp.calculate_mesh_corners(i))
        cv2.polylines(wier, [corners.reshape(1, -1, 2).reshape(-1, 2)], True, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow("test", wier)
    cv2.imwrite("inputs/templates/mesh_WIERpy.png", wier)
    cv2.waitKey()
    cv2.destroyAllWindows()

