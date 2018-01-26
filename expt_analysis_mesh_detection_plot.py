import matplotlib as mtpl
mtpl.use('pdf')
import matplotlib.pylab as plt
plt.switch_backend('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import numpy as np



def a(expt_dir, *args, fn):
     import os
     name, ext = os.path.splitext(fn)
     pp=PdfPages(os.path.join(expt_dir, *args, 'analysis_mesh_deteciont' + name + '.pdf'))
     template_npz = np.load('')

     for key in template_npz.keys():
         a = template_npz[key]
         inline = a[:,:,0]
         plt.figure(figsize=(16,12))
         sns.set("paper", "whitegrid", "dark", font_scale=1.5)
         h=sns.heatmap(inline, annot=True, fmt='d', cmap='Greens')
         h.set(xlabel="Mesh map X")
         h.set(ylabel="Mesh map Y")
         h.set(title="Inline heat map - "+str(key))
         h_fig = h.get_figure()
         h_fig.savefig(pp, format='pdf')

         status = a[:,:,1]
         plt.figure(figsize=(16,12))
         sns.set("paper", "whitegrid", "dark", font_scale=1.5)
         sh=sns.heatmap(status, annot=True, fmt='d', cmap='Oranges')
         sh.set(xlabel="Mesh map X")
         sh.set(ylabel="Mesh map Y")
         sh.set(title="Matched points heat map - "+str(key))
         sh_fig = sh.get_figure()
         sh_fig.savefig(pp, format='pdf')

         ratio = a / status
         plt.figure(figsize=(16,12))
         sns.set("paper", "whitegrid", "dark", font_scale=1.5)
         r=sns.heatmap(status, annot=True, fmt='.3f', cmap='BuPu')
         r.set(xlabel="Mesh map X")
         r.set(ylabel="Mesh map Y")
         r.set(title="Inline Ratio heat map - "+str(key))
         r_fig = r.get_figure()
         r_fig.savefig(pp, format='pdf')

     pp.close()
