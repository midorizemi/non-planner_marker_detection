import numpy as np
import math

offset_c = 100
offset_r = 75
scols = 8
srows = 8


def get_env_setting():
    scaleX = offset_c / offset_c
    scaleY = offset_r / offset_c

    numX = scols + 1
    numY = srows + 1

    return scaleX, scaleY, numX, numY

def get_verts_triangle():
    "__/\__"
    scaleX, scaleY, numX, numY = get_env_setting()
    offset = int(numX / 4)
    centerX = 2 * offset
    centerY = int(numY / 2)
    center_second = centerX - 1
    center_third = centerX + 1
    offset_z = ((offset * scaleX) * math.sin(math.pi / 3)) / 2 #正三角形の内角60度
    offset_x = ((offset * scaleX) * math.cos(math.pi / 3)) / 2 #正三角形の内角60度

    verts = []
    diffX = 0
    diffZ = 0
    for j in range(numX):
        if j == center_second:
            diffX += offset_x
            diffZ = offset_z
        elif j == centerX:
            diffX += offset_x
            diffZ = offset_z * 2
        elif j == center_third:
            diffX += offset_x
            diffZ = offset_z
        else:
            diffZ = 0
        for i in range(numY):
            x = scaleX * j - diffX
            y = scaleY * i
            z = diffZ

            vert = (x, y, z)
            verts.append(vert)

    return verts

def get_face():
    scaleX, scaleY, numX, numY = get_env_setting()
    count = 0
    faces = []
    for i in range(numY * (numX - 1)):
        if count < numY -1:
            A = i
            B = i + 1
            C = (i + numY) + 1
            D = (i + numY)
            face = (A, B, C, D)
            print(face)
            faces.append(face)
        else:
            count = 0
    return faces

if __name__ == "__main__":
    # verts = get_verts_triangle()
    # faces = get_face(verts)
    # mesh = bpy.data.meshes.new('triagle')
    # object = bpy.data.objects.new('triangle', mesh)
    #
    # object.location = [0, 0, 0]
    # bpy.context.scene.objects.link(object)
    #
    # mesh.from_pydata(verts, [], faces)
    # mesh.update(calc_edges=True)
    v = get_verts_triangle()
    f = get_face()
    print(len(v))
    print(len(f))
