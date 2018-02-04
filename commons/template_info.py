class TemplateInfo:
    def __init__(self, _fn="tmp.png", _cols=800, _rows=600, _scols=8, _srows=8, _nneighbor=4, template_img='qrmarker.png'):
        self.fn = _fn
        self.cols = _cols
        self.rows = _rows
        self.scols = _scols
        self.srows = _srows
        self.offset_c = self.cols // self.scols
        self.offset_r = self.rows // self.srows
        self.nneighbor = _nneighbor
        self.tmp_img = template_img

    def get_splitnum(self):
        return self.scols * self.srows

    def make_splitmap(self):
        import numpy as np
        img = np.zeros((self.rows, self.cols, 1), np.uint8)
        for i in range(self.get_splitnum()):
            y, x = self.calculate_mesh_topleftvertex(i)
            mesh = np.tile(np.uint8([i]), (self.offset_r, self.offset_c, 1))
            img[y:y+self.offset_r, x:x+self.offset_c] = mesh

        return img

    def calculate_mesh_topleftvertex(self, id):
        ## coluculate mash top left vertex pt = [Y-axis, X-axis]
        div_r = id // self.scols
        mod_c = id % self.scols

        return  self.offset_r * div_r, self.offset_c * mod_c

    def calculate_mesh_corners_index(self, id):
        return (id + 0, id + 1, id + self.scols, id + self.scols + 1)

    def calculate_mesh_corners(self, id):
        #メッシュ番号を入力メッシュを構成する頂点を返す
        import numpy as np
        i, j = self.calculate_mesh_topleftvertex(id)
        def overw(val):
            if val > self.cols:
                return self.cols -1
            elif val < 0:
                return 0
            else: return val
        def overh(val):
            if val > self.rows:
                return self.rows -1
            elif val < 0:
                return 0
            else: return val
        return np.float32([[i, j], [i, overw(j + self.offset_c)],
                           [overh(i + self.offset_r), overw(j + self.offset_c)], [overh(i + self.offset_r), j]])

    def get_meshid_index(self, id):
        #行列のインデックスを返す
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
        def v_(y, *args):
            a = list(args)
            if int(id/self.scols) == 0:
                #上辺
                a[1] = -1
            if int(id%self.scols) == 0:
                #左辺
                a[0] = -1
            if int(id/self.scols) == self.srows -1:
                #下辺
                a[3] = -1
            if int(id%self.scols) == self.scols -1:
                #サ変
                a[2] = -1
            return a

        if self.nneighbor == 4:
            a = [id - 1, id - self.scols, id + 1, id + self.scols]
            return list(map(validate, v_(id, *a)))
        elif self.nneighbor == 8:
            a = [id - 1, id - 1 - self.scols,
                 id - self.scols, id + 1 - self.scols, id + 1,
                 id + 1 + self.scols, id + self.scols, id + self.scols - 1]
            return list(map(validate, v_(a)))
        elif self.nneighbor == 3:
            """三角メッシュ"""
            if id % 2 == 0:
                """上三角"""
                a = [id - 1, id - (self.scols - 1), id + 1]
                return list(map(validate, v_(a)))
            else:
                """下三角"""
                a = [id - 1, id + 1, id + (self.scols - 1)]
                return list(map(validate, v_(a)))

    def get_nneighbor(self, id, mesh_map=None):
        pass

    def get_mesh_recanglarvertex_list(self, list_id):
        list_mesh_vertex = []
        for id in list_id:
            list_mesh_vertex.append(self.calculate_mesh_corners(id))
        return list_mesh_vertex


    def get_mesh_corners(self, list_merged_mesh_id, merged_map):
        merged_map_lists = []
        for id in list_merged_mesh_id:
            merged_map
        pass
