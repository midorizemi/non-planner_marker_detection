# python

from make_database import split_affinesim as splta
from commons.my_common import format4pickle_pairs
import os

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
        dir_path_full = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
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

    # 実験用出力先パスの生成
    expt_path = splta.myfm.setup_expt_directory(os.path.basename(__file__))
    expt_name = os.path.basename(expt_path)
    testset_name = os.path.basename(testset_dir_full)
    output_dir = splta.myfm.setup_output_directory(expt_name, testset_name)
    detected_dir = splta.myfm.setup_output_directory(output_dir, 'detected_mesh')
    line_dir = splta.myfm.setup_output_directory(output_dir, 'dmesh_line')
    dump_match_dir = splta.myfm.setup_output_directory(output_dir, 'dump_match_dir')
    dump_detected_dir = splta.myfm.setup_output_directory(output_dir, 'dump_detected_dir')

    fn, ext = os.path.splitext(os.path.basename(fn2_full))
    imgT = splta.cv2.imread(fn2_full, 0)
    if imgT is None:
        print('Failed to load fn2:', fn2_full)
        sys.exit(1)
    print("INPUT: {}".format(fn2_full))

    pool = splta.ThreadPool(processes=splta.cv2.getNumberOfCPUs())
    with splta.Timer('Detection'):
        kpT, descT = splta.affine_detect(detector, imgT, pool=pool, simu_param='test')
    print('imgQ - %d features, imgT - %d features' % (splta.count_keypoints(splt_kpQ), len(kpT)))

    with splta.Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)
    index_mesh_pairs = format4pickle_pairs(mesh_pairs)
    import joblib

    dump_match_testcase_dir = splta.myfm.setup_output_directory(dump_match_dir, fn)
    print(dump_match_testcase_dir)
    joblib.dump(mesh_pQ, os.path.join(dump_match_testcase_dir, 'mesH_pQ.pikle'), compress=True)
    joblib.dump(mesh_pT, os.path.join(dump_match_testcase_dir, 'mesH_pT.pikle'), compress=True)
    import pickle

    with open(os.path.join(dump_match_testcase_dir, 'mesh_pairs.pickle'), 'wb') as f:
        pickle.dump(index_mesh_pairs, f)
        f.close()

    with splta.Timer('estimation'):
        Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)
    joblib.dump(Hs, os.path.join(dump_match_testcase_dir, 'Hs.pikle'), compress=True)
    joblib.dump(statuses, os.path.join(dump_match_testcase_dir, 'statuses.pikle'), compress=True)
    index_pairs = tuple(
        tuple((p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in pair) for pair in pairs)
    with open(os.path.join(dump_match_testcase_dir, 'pairs.pickle'), 'wb') as f:
        pickle.dump(index_pairs, f)

    # vis = splta.draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    # splta.cv2.imwrite(os.path.join(detected_dir, fn + '.png'), vis)
    #
    # viw = splta.explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs, Hs=Hs)
    #
    # splta.cv2.imwrite(os.path.join(line_dir, fn + '.png'), viw)
    # splta.cv2.destroyAllWindows()
