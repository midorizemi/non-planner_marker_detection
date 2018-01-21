class TemplateInfo:
    def __init__(self, _fn="tmp.png", _cols=800, _rows=600, _scols=8, _srows=8, _nneighbor=4):
        self.fn = _fn
        self.cols = _cols
        self.rows = _rows
        self.scols = _scols
        self.srows = _srows
        self.offset_c = self.cols // self.scols
        self.offset_r = self.rows // self.srows
        self.nneighbor = _nneighbor

    def get_splitnum(self):
        return self.scols * self.srows

    ## coluculate mash top left vertex pt
    def calculate_mesh_tlvertex(self, id):
        div_r = id // self.scols
        mod_c = id % self.scols

        return self.offset_c * mod_c, self.offset_r * div_r

    def calculate_mesh_corners_index(self, id):
        return (id + 0, id + 1, id + self.scols, id + self.scols + 1)

    def calculate_mesh_corners(self, id):
        import numpy as np
        i, j = self.calculate_mesh_tlvertex(id)
        return np.float32(
            [[i, j], [i + self.offset_c, j], [i + self.offset_c, j + self.offset_r], [i, j + self.offset_r]])

    def get_meshid_vertex(self, id):
        div_r = id // self.scols
        mod_c = id % self.scols
        return div_r, mod_c

    def get_mesh_map(self):
        import numpy as np
        mesh_ids = np.arange(self.get_splitnum()).reshape(self.srows, self.scols)
        return mesh_ids

    def get_mesh_shape(self):
        return self.srows, self.scols

    def get_meshidlist_nneighbor(self, id):
        """
        直線リストにおける メッシュIDの近傍に属するIDを取得する
        :param id:
        :return:
        """

        def validate(x):
            if x < 0:
                return None
            else:
                return x

        if self.nneighbor == 4:
            a = [id - 1, id - self.scols, id + 1, id + self.scols]
            return list(map(validate, a))
        elif self.nneighbor == 8:
            a = [id - 1, id - 1 - self.scols,
                 id - self.scols, id + 1 - self.scols, id + 1,
                 id + 1 + self.scols, id + self.scols, id + self.scols - 1]
            return list(map(validate, a))
        elif self.nneighbor == 3:
            """三角メッシュ"""
            if id % 2 == 0:
                """上三角"""
                a = [id - 1, id - (self.scols - 1), id + 1]
                return list(map(validate, a))
            else:
                """下三角"""
                a = [id - 1, id + 1, id + (self.scols - 1)]
                return list(map(validate, a))

    def get_nneighbor(self, id, mesh_map=None):
        pass
