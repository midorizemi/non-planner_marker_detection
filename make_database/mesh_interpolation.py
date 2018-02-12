from make_database import split_affinesim as splta
from commons.my_common import load_pickle_mesh_matchepairs
from commons.my_common import load_pickle_matchepairs
import math
import os


def load_pickle_match_with_cross(expt_name, testset_name, fn):
    output_dir = splta.myfm.setup_output_directory(expt_name, testset_name)
    dump_match_dir = splta.myfm.setup_output_directory(output_dir, 'dump_match_dir')
    dump_match_testcase_dir = splta.myfm.setup_output_directory(dump_match_dir, fn)
    import joblib
    mesh_pQ = joblib.load(os.path.join(dump_match_testcase_dir, 'mesH_pQ.pikle'))
    mesh_pT = joblib.load(os.path.join(dump_match_testcase_dir, 'mesH_pT.pikle'))
    each_mesh_matchnum = list(len(mesh) for mesh in mesh_pQ)
    mesh_pairs = load_pickle_mesh_matchepairs(os.path.join(dump_match_testcase_dir, 'mesh_pairs.pickle'),
                                              each_mesh_matchnum)
    return mesh_pQ, mesh_pT, mesh_pairs


def load_pickle_calclate_Homography4splitmesh(expt_name, testset_name, fn):
    output_dir = splta.myfm.setup_output_directory(expt_name, testset_name)
    dump_match_dir = splta.myfm.setup_output_directory(output_dir, 'dump_match_dir')
    dump_match_testcase_dir = splta.myfm.setup_output_directory(dump_match_dir, fn)
    import joblib
    Hs = joblib.load(os.path.join(dump_match_testcase_dir, 'Hs.pikle'))
    statuses = joblib.load(os.path.join(dump_match_testcase_dir, 'statuses.pikle'))
    pairs = load_pickle_matchepairs(os.path.join(dump_match_testcase_dir, 'pairs.pickle'))
    return Hs, statuses, pairs


def explore_meshes(imgT=None, Hs=None, temp_inf=None):
    if imgT is not None:
        hT, wT = imgT.shape[:2]
    else:
        hT, wT = 0, 0
    if temp_inf is None:
        temp_inf = splta.TmpInf()
    meshes = []
    gHs = []
    for id, H in enumerate(Hs):
        mesh_corners = temp_inf.calculate_mesh_corners(id)
        if H is not None:
            corners = splta.np.int32(splta.cv2.perspectiveTransform(mesh_corners.reshape(1, -1, 2), H).reshape(-1, 2)
                                     + (wT, 0))
            if is_goodMeshEstimation(corners, temp_inf):
                meshes.append(corners)
                gHs.append(H)
            else:
                meshes.append(None)
                gHs.append(None)
        else:
            meshes.append(None)
            gHs.append(None)

    return meshes, gHs


def is_goodMeshEstimation(corners, temp_inf):
    hole_area = temp_inf.cols * temp_inf.rows
    mesh_area = temp_inf.offset_c * temp_inf.offset_r
    pt1 = corners[0]
    pt2 = corners[1]
    pt3 = corners[2]
    pt4 = corners[3]
    vect13 = pt3 - pt1
    vect24 = pt4 - pt2

    norm_13MUL24 = splta.np.linalg.norm(vect13) * splta.np.linalg.norm(vect24)
    if norm_13MUL24 <= 0.1:
        return False
    co = splta.np.dot(vect13, vect24)
    sinTheta = splta.np.sin(splta.np.arccos(co / norm_13MUL24))
    local_area = norm_13MUL24 / 2 * sinTheta
    if not hole_area / 2 > local_area or not mesh_area / 10 < local_area:
        # 質が悪いやつ
        return False

    def is_cross(p1, p2, p3, p4):
        t1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        if t1 < 0:
            a1 = -1
        else:
            a1 = 1
        t2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        if t2 < 0:
            a2 = -1
        else:
            a2 = 1
        t3 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        if t3 < 0:
            a3 = -1
        else:
            a3 = 1
        t4 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
        if t4 < 0:
            a4 = -1
        else:
            a4 = 1
        return a1 * a2 < 0 and a3 * a4 < 0

    if is_cross(pt1, pt2, pt3, pt4):
        return False
    if is_cross(pt2, pt3, pt4, pt1):
        return False
    if not is_cross(pt1, pt3, pt4, pt2):
        return False
    return True


def draw_matches_for_meshes(imgT, imgQ, Hs=None, vis=None, estimated=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    if vis is None:
        vis = splta.np.zeros((max(h1, h2), w1 + w2), splta.np.uint8)
        vis[:h1, :w1] = imgT
        vis[:h2, w1:w1 + w2] = imgQ
        vis = splta.cv2.cvtColor(vis, splta.cv2.COLOR_GRAY2BGR)
    meshes, Hss = explore_meshes(imgT, Hs)
    for corners in meshes:
        if corners is None:
            continue
        splta.cv2.polylines(vis, [corners], True, (255, 255, 0), thickness=3, lineType=splta.cv2.LINE_AA)
    if estimated is not None:
        for e in estimated:
            splta.cv2.polylines(vis, [meshes[e]], True, (10, 100, 255), thickness=3, lineType=splta.cv2.LINE_AA)

    return vis


def is_detectable(lenmesh, median):
    if not median == 0:
        threshold = median * 0.01
        if threshold < 4:
            threshold = 4
    else:
        threshold = 4
    if lenmesh >= threshold:
        return True
    else:
        return False


def calculate_centerofgrabity_position(origin, goodvertexes):
    o_set = splta.np.unique(origin, axis=0).tolist()
    g_corner = splta.np.empty((0, 2), splta.np.float32)
    o_corner = splta.np.empty((0, 2), splta.np.float32)
    for oset in o_set:
        corner_index = splta.np.where(origin == oset[0])
        a_ = origin[corner_index[0]]
        a_index = splta.np.where(a_ == oset[1])
        vertexes = goodvertexes[corner_index[0][a_index[0]]]
        if vertexes.shape[0] == 2:
            v_ = splta.np.array([splta.np.sum(vertexes[:, 0]) / 2, splta.np.sum(vertexes[:, 1]) / 2],
                                splta.np.float32)
        elif vertexes.shape[0] == 1:
            v_ = splta.np.float32(vertexes[0])
        else:
            v_ = splta.np.array([splta.np.sum(vertexes[:, 0]) / 3, splta.np.sum(vertexes[:, 1]) / 3],
                                splta.np.float32)
        g_corner = splta.np.append(g_corner, v_.reshape(1, 2), axis=0)
        v_ = splta.np.array(oset, splta.np.float32)
        o_corner = splta.np.append(o_corner, v_.reshape(1, 2), axis=0)

    return o_corner, g_corner


def extract_mesh_vertexes(map_goodmesh, neghibors_8, _mesh_corners, temp_inf):
    goodid = list(i for i in neghibors_8 if i is not None if map_goodmesh[temp_inf.get_meshid_index(i)])
    goodvertexes = splta.np.empty((0, 2), splta.np.int32)
    origin = splta.np.empty((0, 2), splta.np.int32)
    for gi in goodid:
        goodvertexes = splta.np.append(goodvertexes, _mesh_corners[gi], axis=0)
        origin = splta.np.append(origin, temp_inf.calculate_mesh_corners(gi), axis=0)

    return origin, goodvertexes


def test_bug_out_img(imgQ, imgT, temp_inf, mesh_corners,
                     map_goodmesh, bi, n8, origin, good_vertexes, o_corner, g_corner, intermediate_dir):
    global gi
    h1, w1 = imgQ.shape[:2]
    h2, w2 = imgT.shape[:2]
    vis = splta.np.zeros((max(h1, h2), w1 + w2), splta.np.uint8)
    vis[:h1, :w1] = imgQ
    vis[:h2, w1:w1 + w2] = imgT
    vis = splta.cv2.cvtColor(vis, splta.cv2.COLOR_GRAY2BGR)
    for i in range(origin.shape[0]):
        splta.cv2.arrowedLine(vis, tuple(splta.np.int32(origin[i].tolist())),
                              tuple(splta.np.int32(good_vertexes[i] + (w1, 0)).tolist()), (0, 190, 255), thickness=2)
    goodid = list(i for i in n8 if i is not None if map_goodmesh[temp_inf.get_meshid_index(i)])
    for gi in goodid:
        c = mesh_corners[gi] + (w1, 0)
        oc = splta.np.int32(temp_inf.calculate_mesh_corners(gi))
        splta.cv2.polylines(vis, [c], True, (0, 255, 0), thickness=3, lineType=splta.cv2.LINE_AA)
        splta.cv2.polylines(vis, [oc], True, (0, 0, 255), thickness=3, lineType=splta.cv2.LINE_AA)
    for i in range(o_corner.shape[0]):
        splta.cv2.circle(vis, tuple(splta.np.int32(o_corner[i].tolist())), 3, (255, 255, 0), thickness=-1)
        splta.cv2.circle(vis, tuple(splta.np.int32(g_corner[i] + (w1, 0)).tolist()), 3, (255, 255, 0), thickness=-1)
        splta.cv2.arrowedLine(vis, tuple(splta.np.int32(o_corner[i].tolist())),
                              tuple(splta.np.int32(g_corner[i] + (w1, 0)).tolist()), (255, 0, 0), thickness=2,
                              tipLength=0.05)
    splta.cv2.imwrite(os.path.join(intermediate_dir, 'points_{}.png'.format(bi)), vis)
    splta.cv2.waitKey(1)
    splta.cv2.destroyAllWindows()


def interpolate_mesh(_denied_num, _Hs, _temp_inf, output=False, intermediate_dir="", imgQ=None, imgT=None):
    mesh_vertexes, g_Hs = explore_meshes(Hs=_Hs)
    map_goodmesh = splta.np.array(list(True if corners is not None else False for corners in mesh_vertexes))
    map_goodmesh = map_goodmesh.reshape(_temp_inf.srows, _temp_inf.srows)
    map_mesh = _temp_inf.get_mesh_map()
    good_area = map_mesh[map_goodmesh].tolist()
    bad = []
    for gi in good_area:
        neighbor_list = _temp_inf.get_meshidlist_nneighbor(gi)
        tmp = list(i for i in neighbor_list if i is not None if not map_goodmesh[_temp_inf.get_meshid_index(i)])
        bad.extend(tmp)
    bad = list(set(bad))
    bad = sorted(bad)
    estimated_mesh = []
    while True:
        for bi in bad:
            n8 = _temp_inf.get_meshidlist_8neighbor(bi)
            origin, good_vertexes = extract_mesh_vertexes(map_goodmesh, n8, mesh_vertexes, _temp_inf)
            o_corner, g_corner = calculate_centerofgrabity_position(origin, good_vertexes)

            if output:
                test_bug_out_img(imgQ, imgT, _temp_inf, mesh_vertexes,
                                 map_goodmesh=map_goodmesh, bi=bi, n8=n8, origin=origin,
                                 good_vertexes=good_vertexes, o_corner=o_corner,
                                 g_corner=g_corner, intermediate_dir=intermediate_dir)
            H, status = splta.cv2.findHomography(o_corner, g_corner, splta.cv2.LMEDS)
            if status is not None and not len(status) == 0:
                print("{0} / {1} = {2:0.3f} inliers/matched=ratio".format(splta.np.sum(status), len(status),
                                                                          splta.np.sum(status) / len(status)))
                # do not draw outliers (there will be a lot of them)
            origin_corners = _temp_inf.calculate_mesh_corners(bi)
            if H is not None:
                corners = splta.np.int32(splta.cv2.perspectiveTransform(origin_corners.reshape(1, -1, 2),
                                                                        H).reshape(-1, 2))
                if is_goodMeshEstimation(corners, _temp_inf):
                    estimated_mesh.append(bi)
                    mesh_vertexes[bi] = corners
                    g_Hs[bi] = H
        dab = []
        for bi in bad:
            if mesh_vertexes[bi] is None:
                dab.append(bi)
                continue
            neighbor_list = _temp_inf.get_meshidlist_nneighbor(bi)
            tmp = list(i for i in neighbor_list if i is not None if not map_goodmesh[_temp_inf.get_meshid_index(i)])
            dab.extend(tmp)
        dab = list(set(dab))
        dab = sorted(dab)
        bad = dab
        if (_denied_num <= len(bad)):
            break
    estimated_mesh = list(set(estimated_mesh))
    estimated_mesh = sorted(estimated_mesh)
    return mesh_vertexes, g_Hs, estimated_mesh


def get_nodes_has_positions(*args, mesh_corners):
    n0, n1, n2, n3 = args
    v0, v1, v2, v3 = None, None, None, None
    if n0 is not None:
        v0 = mesh_corners[n0][0]
    if n1 is not None:
        v1 = mesh_corners[n1][1]
    if n2 is not None:
        v2 = mesh_corners[n2][2]
    if n3 is not None:
        v3 = mesh_corners[n3][3]
    return v0, v1, v2, v3

def get_nodes_dispersion(*args):
    v0, v1, v2, v3 = args
    l01, l02, l03, l12, l13, l23 = 0, 0, 0, 0, 0, 0
    p = 0
    if not (v0 is None or v1 is None):
        l01 = splta.np.linalg.norm(v1 - v0)
        p += 1
    if not (v0 is None or v2 is None):
        l02 = splta.np.linalg.norm(v2 - v0)
        p += 1
    if not (v0 is None or v3 is None):
        l03 = splta.np.linalg.norm(v3 - v0)
        p += 1
    if not (v1 is None or v2 is None):
        l12 = splta.np.linalg.norm(v2 - v1)
        p += 1
    if not (v1 is None or v3 is None):
        l13 = splta.np.linalg.norm(v3 - v1)
        p += 1
    if not (v2 is None or v3 is None):
        l23 = splta.np.linalg.norm(v3 - v2)
        p += 1
    if p == 0:
        return None
    else:
        W = 1/p * (l01 + l02 + l03 + l12 + l13 + l23)
    height, width = imgQ.shape
    S = height * width
    #一様ポアソン分布
    Ew = 1/(2 * math.sqrt(p/S))
    w = W / Ew
    return w

if __name__ == '__main__':
    print(__doc__)

    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1_full, fn2_full, pr, testset_dir_full = args
    except:
        dir_path_full = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        fn1_full = os.path.abspath(os.path.join(dir_path_full, 'data/templates/qrmarker.png'))
        fn2_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker/057_070-200.png'))
        testset_dir_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker'))
        pr = "mltf_"

    imgQ = splta.cv2.imread(fn1_full, 0)
    detector, matcher = splta.init_feature(feature_name)

    if imgQ is None:
        print('Failed to load fn1:', fn1_full)
        sys.exit(1)
    print("MARKER: {}".format(fn1_full))

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    print("FEATURE: {}".format(feature_name))

    scols = 8
    srows = 8
    h, w = imgQ.shape[:2]
    template_fn, ext = os.path.splitext(os.path.basename(fn1_full))
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": w, "_rows": h, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    temp_inf = splta.TmpInf(**template_information)

    try:
        with splta.Timer('Lording pickle'):
            splt_kpQ, splt_descQ = splta.affine_load_into_mesh(template_fn, temp_inf.get_splitnum())
    except ValueError as e:
        print(e.args)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute makedb/make_split_combine_featureDB_from_templates.py')
        with splta.Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = splta.affine_detect_into_mesh(detector, temp_inf.get_splitnum(),
                                                                 imgQ, simu_param='asift')

    mesh_k_num = splta.np.array([len(keypoints) for keypoints in splt_kpQ]).reshape(temp_inf.get_mesh_shape())
    median = splta.np.nanmedian(mesh_k_num)

    fn, ext = os.path.splitext(os.path.basename(fn2_full))
    testset_name = os.path.basename(os.path.dirname(fn2_full))
    imgT = splta.cv2.imread(fn2_full, 0)
    if imgT is None:
        print('Failed to load fn2:', fn2_full)
        sys.exit(1)
    print("INPUT: {}".format(fn2_full))

    pool = splta.ThreadPool(processes=splta.cv2.getNumberOfCPUs())
    with splta.Timer('Detection'):
        kpT, descT = splta.affine_detect(detector, imgT, pool=pool, simu_param='test')
    print('imgQ - %d features, imgT - %d features' % (splta.count_keypoints(splt_kpQ), len(kpT)))

    dumped_exdir = "expt_split_affinesim"
    # dumped_exdir = "expt_split_affinesim_conbine"
    try:
        with splta.Timer('Loarding matching pickle'):
            mesh_pQ, mesh_pT, mesh_pairs = load_pickle_match_with_cross(dumped_exdir, testset_name, fn)
            # mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)
    except:
        print('Failed Load matching result')
        with splta.Timer('matching'):
            mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)

    try:
        with splta.Timer('Loading estimation result'):
            Hs, statuses, pairs = load_pickle_calclate_Homography4splitmesh(dumped_exdir, testset_name, fn)
            # Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)
    except:
        print('Failed loading estimated mesh')
        with splta.Timer('estimation'):
            Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)

    # 検出不可能メッシュ
    denied_mesh = list(is_detectable(len(match), median) for match in mesh_pQ)
    denied_num = len(denied_mesh) - sum(denied_mesh)
    mesh_corners, good_Hs, estimated = interpolate_mesh(denied_num, Hs, temp_inf)

    # viw = splta.explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs, Hs=Hs)
    # viw = splta.draw_matches_for_meshes(imgQ=imgT, imgT=imgQ, Hs=Hs)
    # splta.cv2.imshow('test',viw)
    # splta.cv2.imwrite('tset.png', viw)
    # splta.cv2.waitKey()

    # viw = draw_matches_for_meshes(imgQ, imgT, Hs=good_Hs, vis=None, estimated=estimated)
    # # viw = draw_matches_for_meshes(imgQ, imgT, Hs=good_Hs, vis=None, estimated=None)
    # splta.cv2.imshow('test', viw)
    # splta.cv2.imwrite('goodMesh.png', viw)
    # splta.cv2.waitKey(1)
    # # viw = splta.draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    # # splta.cv2.imshow('test', viw)
    # # splta.cv2.imwrite('estimated.png', viw)
    # # splta.cv2.waitKey()
    # splta.cv2.destroyAllWindows()

    nodes_has_meshes = temp_inf.get_nodes_has_meshes_id()
    nodes_has_posisions = list(get_nodes_has_positions(*m, mesh_corners=mesh_corners) for m in nodes_has_meshes)
    nodes_dispersion = list(get_nodes_dispersion(*vs) for vs in nodes_has_posisions)
    print(nodes_dispersion)
