class TemplateInfo:
    def __init__(self, _fn="tmp.png", _cols=800, _rows=600, _scols=8, _srows=8):
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
    def calculate_mesh_tlvertex(self, id):
        div_r = id // self.scols
        mod_c = id % self.scols

        return self.offset_c*mod_c, self.offset_r*div_r

    def calculate_mesh_corners_index(self, id):
        return (id+0, id+1, id+self.scols, id+self.scols+1)

    def calculate_mesh_corners(self, id):
        import numpy as np
        i, j = self.calculate_mesh_tlvertex(id)
        return np.float32([[i, j], [i + self.offset_c, j], [i + self.offset_c, j + self.offset_r], [i, j + self.offset_r]])
