import matplotlib as mtpl
# mtpl.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as plt
#plt.switch_backend('pdf')
import seaborn as sns

import numpy as np

from typing import List
def a_params()-> List[List]:
    """一周する"""
    params = [[1, 0]]
    for t in np.reciprocal(np.cos(np.radians(np.arange(10, 90, 10)))):
        for phi in np.radians(np.arange(0, 360, 10)):
            params.append([phi, t])
    return params

def plot_mesh(mesh, key, plot_title, color):
    plt.figure(figsize=(16,12))
    sns.set("paper", "whitegrid", "dark", font_scale=1.5)
    fmt = 'f' if mesh.dtype == np.float64 else 'd'
    h = sns.heatmap(mesh, annot=True, fmt=fmt, cmap=color)
    h.set(xlabel="Mesh map X")
    h.set(ylabel="Mesh map Y")
    h.set(title=plot_title + " heat map - "+key)
    return h

def test_data():
    expt_dir = "/home/tiwasaki/PycharmProjects/data/outputs/expt_mesh_detection_performance_spltASIFT"
    fn = "pl_qrmarker.npz"
    return expt_dir, fn

def plot_each_cam_position(expt_dir, fn):
    import os
    name, ext = os.path.splitext(fn)
    pp=PdfPages(os.path.join(expt_dir, 'analysis_mesh_deteciont' + name + '.pdf'))
    template_npz = np.load(os.path.join(expt_dir, fn))

    #[phi, t]のリスト
    params = a_params()
    for i, key in enumerate(template_npz.keys()):
        a = template_npz[key]
        #a (8 * 8 * 3)の各メッシュ内の特徴点数の行列
        #a[:, :, 0] =>メッシュ内のHomographyで求めた誤差範囲内のキーポイント数
        #a[:, :, 1] =>メッシュ内のマッチングポイント数全て
        h = plot_mesh(a[:, :, 0], key, "Refine points", 'Blues')
        h_fig = h.get_figure()
        h_fig.savefig(pp, format='pdf')

        sh = plot_mesh(a[:, :, 1], key, "Matched points", 'Reds')
        sh_fig = sh.get_figure()
        sh_fig.savefig(pp, format='pdf')

        ratio = a[:, :, 0] / a[:, :, 1]
        rh = plot_mesh(ratio, key, "Ratio of refine and matched points \n", 'Purples')
        r_fig = rh.get_figure()
        r_fig.savefig(pp, format='pdf')

        #メッシュ毎の平均特徴点数
        means = list(np.mean(s) for s in [a[:, :, 0], a[:, :, 1], ratio])
        #マッチングできた数
        sums = list(np.sum(s) for s in [a[:, :, 0], a[:, :, 1]])
        #メッシュの数
        mesh_num = np.count_nonzero(a[:, :, 1])

        #params= [phi, t, sum_refine, sum_match, mean_refine, mean_match, mean_ratio, mesh_num]
        params[i].extend(sums)
        params[i].extend(means)
        params[i].append(mesh_num)
    pp.close()
    return params

def plot_detection_passage(expt_dir, *args, fn, params):
    import os
    name, ext = os.path.splitext(fn)
    pp = PdfPages(os.path.join(expt_dir, *args, 'detection_analysis_via_cam_position_passage' + name + '.pdf'))

    phi_ticklabels = [r"${}^\circle$".format(phi) for phi in np.arange(0, 360, 10)]
    t_rgrids = [r"${}^\circle$".format(t) for t in np.arange(90, 10, 10)]
    columns = ["longitude", "latitude", "sum_refine", "sum_match", "mean_refine", "mean_matc", "mean_ratio", "mesh_num"]


if __name__ == '__main__':
    expt_dir, fn = test_data()
    params = plot_each_cam_position(expt_dir, fn)
