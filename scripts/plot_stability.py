import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from solution_linear import solution_linear
import numpy as np
from scipy.sparse import linalg

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call


if __name__ == "__main__":

    N_re = 10
    N_im = 10
    lam_im_max = 50.0
    lam_re_max =  1.0
    lambda_im = 1j*np.linspace(0.0, lam_im_max, N_im)
    lambda_re = np.linspace(-lam_re_max, 1.0, N_re)

    nfine   = 10
    ncoarse = 1
    niter   = 3
    nslices = 100
    stab    = np.zeros((N_im, N_re), dtype='complex')
    for i in range(0,N_re):
      for j in range(0,N_im):
        u0 = solution_linear(y = np.array( [[1.0]], dtype='complex'),  A = np.array([[lambda_re[i]+lambda_im[j]]],dtype='complex'))
        para = parareal(0.0, 1.0, nslices, impeuler, impeuler, ncoarse, nfine, 0.0, niter, u0)
        Mat = para.get_parareal_stab_function(niter)
        stab[j,i] = abs(Mat)
    ###
    #rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    #pcol = plt.pcolor(lambda_s.imag, lambda_f.imag, np.absolute(stab), vmin=0.99, vmax=2.01)
    #pcol.set_edgecolor('face')
    levels = np.array([0.25, 0.5, 0.75, 0.9, 1.1])
#    levels = np.array([1.0])
    CS1 = plt.contour(lambda_re, lambda_im.imag, np.absolute(stab), levels, colors='k', linestyles='dashed')
    CS2 = plt.contour(lambda_re, lambda_im.imag, np.absolute(stab), [1.0],  colors='k')
    plt.clabel(CS1, inline=True, fmt='%3.2f', fontsize=fs-2)
    manual_locations = [(1.5, 2.5)]
    plt.clabel(CS2, inline=True, fmt='%3.2f', fontsize=fs-2, manual=manual_locations)
    #plt.gca().add_patch(Polygon([[0, 0], [lam_s_max,0], [lam_s_max,lam_s_max]], visible=True, fill=True, facecolor='.75',edgecolor='k', linewidth=1.0,  zorder=11))
    #plt.plot([0, 2], [0, 2], color='k', linewidth=1, zorder=12)
    #plt.gca().set_xticks(np.arange(0, int(lam_s_max)+1))
    #plt.gca().set_yticks(np.arange(0, int(lam_f_max)+2, 2))
    #plt.gca().tick_params(axis='both', which='both', labelsize=fs)
    #plt.xlim([0.0, lam_s_max])
    #plt.ylim([0.0, lam_f_max])
    #plt.xlabel('$\Delta t \lambda_{slow}$', fontsize=fs, labelpad=2.0)
    #plt.ylabel('$\Delta t \lambda_{fast}$', fontsize=fs)
    #plt.title(r'$M=%1i$, $K=%1i$' % (swparams['num_nodes'],K), fontsize=fs)
    #plt.show()
    #filename = 'sdc-fwsw-stability-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    #fig.savefig(filename, bbox_inches='tight')
    #call(["pdfcrop", filename, filename])
    plt.show()
