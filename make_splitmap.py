# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

class template:
    def __init__(self, _fn, _cols=800, _rows=600, _scols=8, _srows=8):
        self.fn = _fn
        self.cols = _cols
        self.rows = _rows
        self.scols = _scols
        self.srows = _srows
        self.offset_c = self.cols//self.scols
        self.offset_r = self.rows//self.srows

    def get_splitnum(self):
        return self.scols*self.srows

    ## coluculate mash top left vertex pt
    def colculate_mesh_tlvertex(self, id):
        div_r = id // self.scols
        mod_c = id % self.cols

        return self.offset_c*mod_c, self.offset_r*div_r




def make_splitmap(split_tmp):
    assert isinstance(split_tmp, template)
    img = np.zeros((split_tmp.rows, split_tmp.cols, 1), np.uint8)
    for i in range(split_tmp.get_splitnum()):
        x, y = split_tmp.colculate_mesh_tlvertex(i)
        mesh = np.tile(np.uint8([i]), (split_tmp.offset_r, split_tmp.offset_c, 1))
        img[x:split_tmp.offset_r, y:split_tmp.offset_c] = mesh

    return img


if __name__ == '__main__':
    tmp = template("inputs/templates/mesh_labelpy.png")
    tmp_img = make_splitmap(tmp)
    cv2.imshow("test", tmp_img)

