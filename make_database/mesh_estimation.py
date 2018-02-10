
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
        dir_path_full = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        fn1_full = os.path.abspath(os.path.join(dir_path_full, 'data/templates/qrmarker.png'))
        fn2_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/unittest/smpl_1.414214_152.735065.png'))
        testset_dir_full = "pl_qrmarker"

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

    fn, ext = os.path.splitext(fn2_full)
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
