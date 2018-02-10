
from make_database import split_affinesim as splta
from commons.my_common import load_pickle_mesh_matchepairs
from commons.my_common import load_pickle_matchepairs
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

def explore_meshes(Hs=None):
    meshes = []
    for id, H in enumerate(Hs):
        mesh_corners = temp_inf.calculate_mesh_corners(id)
        if H is not None:
            corners = splta.np.int32(splta.cv2.perspectiveTransform(mesh_corners.reshape(1, -1, 2), H).reshape(-1, 2))
            if is_goodMeshEstimation(corners):
                meshes.append(corners)
            else:
                meshes.append(None)
                Hs[id] = None
        else:
            meshes.append(None)

    return meshes, Hs

def is_goodMeshEstimation(corners: splta.np.ndarray):
    hole_area = temp_inf.cols * temp_inf.rows
    mesh_area = temp_inf.offset_c * temp_inf.offset_r
    pt1 = corners[0]
    pt2 = corners[1]
    pt3 = corners[2]
    pt4 = corners[3]
    vect13 = pt3 - pt1
    vect24 = pt4 - pt2

    leng13 = splta.np.linalg.norm(vect13)
    leng24 = splta.np.linalg.norm(vect24)
    sinTheta = splta.np.sin(splta.np.arccos(vect13.dot(vect24)/leng13*leng24))
    local_area = leng13*leng24/2*sinTheta
    if not hole_area/2 > local_area or not mesh_area/10 < local_area:
        #質が悪いやつ
        return False

    def is_cross(p1, p2, p3, p4):
        t1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        t2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        t3 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        t4 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
        return t1 * t2 < 0 and t3 * t4 < 0
    if is_cross(pt1, pt2, pt3, pt4):
        return False
    if is_cross(pt2, pt3, pt4, pt1):
        return False
    return True




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
    imgT = splta.cv2.imread(fn2_full, 0)
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
    w, h = imgQ.shape[:2]
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
    #dumped_exdir = "expt_split_affinesim_conbine"
    try:
        with splta.Timer('Loarding matching pickle'):
            mesh_pQ, mesh_pT, mesh_pairs = load_pickle_match_with_cross(dumped_exdir, testset_name, fn)
    except :
        print('Failed Load matching result')
        with splta.Timer('matching'):
            mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)

    try:
        with splta.Timer('Loading estimation result'):
            Hs, statuses, pairs = load_pickle_calclate_Homography4splitmesh(dumped_exdir, testset_name, fn)
    except:
        print('Failed loading estimated mesh')
        with splta.Timer('estimation'):
            Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)

    mesh_corners, good_Hs = explore_meshes(Hs)

    print("xxxx")