import numpy as np
import math
from commons.template_info import TemplateInfo as TmpInf

temp_inf = TmpInf()

def get_env_setting(_tmp: TmpInf):
    scaleX = _tmp.offset_c / _tmp.offset_c
    scaleY = _tmp.offset_r / _tmp.offset_c

    numX = _tmp.scols + 1
    numY = _tmp.srows + 1

    return scaleX, scaleY, numX, numY

def get_verts(_tmp: TmpInf):
    "__/\__"
    scaleX, scaleY, numX, numY = get_env_setting(_tmp)
    offset = int(numX / 4)
    centerX = 2 * offset
    centerY = int(numY / 2)
    center_second = centerX - 1
    center_third = centerX + 1
    offset_z = ((offset * scaleX) * math.sin(math.pi / 3)) / 2 #正三角形の内角60度
    offset_x = ((offset * scaleX) * math.cos(math.pi / 3)) / 2 #正三角形の内角60度

    verts = []
    for i in range(numY):
        diffX = 0
        diffZ = 0
        for j in range(numX):
            if j == center_second or j == centerX or j == center_third:
                diffX += offset_x
                diffZ += offset_z
            x = scaleX * j - centerX * scaleX
            y = scaleY * i - centerY * scaleY
            z = diffZ
            vert = (x, y, z)
            verts.append(vert)

    return verts

def get_face(verts, _tmp: TmpInf):
    scaleX, scaleY, numX, numY = get_env_setting(_tmp)
    count = 0
    faces = []
    for i in range(numX * (numY - 1)):
        if count < numX -1:
            A = i
            B = i + 1
            C = (1 + numX) + 1
            D = (1 + numX)
            face = (A, B, C, D)
            faces.append(face)
        else:
            count = 0






