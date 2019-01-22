# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:33:10 2019

@author: FMA
"""

import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import functions as f

data = scipy.io.loadmat('data.mat')
xdisc = data.get('xdisc')
xunif = data.get('xunif')

#plt.figure()
#plt.title('xdisc data representation')
#plt.xlabel('Dimension 1')
#plt.ylabel('Dimension 2')
#plt.plot(xdisc[0,:], xdisc[1,:], 'o')
#plt.show()
#
#plt.figure()
#plt.title('xunif data representation')
#plt.xlabel('Dimension 1')
#plt.ylabel('Dimension 2')
#plt.plot(xunif[0,:], xunif[1,:], 'o')
#plt.show()


"""2D Kohonen Map"""
K = 4
n = 2

nbiter = 10000

nbr_affichage = 10


sigma_max = 3.0
sigma_min = 0.1
mu_max = 0.5
mu_min = 0.1

Wp_xunif = f.kohonen2d(xunif, K, mu_max, mu_min, sigma_max, sigma_min, nbiter, nbr_affichage, 'xunif')

titre = 'Final Result Kohonen 2D xunif'
f.affiche_grille_final(Wp_xunif, xunif, titre) 

#Classification
clas_xunif = f.clas_Kohonen2D(xunif, K, Wp_xunif)

#Display classification

f.affiche_clas_kohonen2D(xunif, clas_xunif, Wp_xunif, 'xunif data set')

Wp_xdisc = f.kohonen2d(xdisc, K, mu_max, mu_min, sigma_max, sigma_min, nbiter, nbr_affichage, 'xdisc')

titre = 'Final Result Kohonen 2D xdisc'
f.affiche_grille_final(Wp_xdisc, xdisc, titre)


#Classification
clas_xdisc = f.clas_Kohonen2D(xdisc, K, Wp_xunif)

#Display classification

f.affiche_clas_kohonen2D(xdisc, clas_xdisc, Wp_xdisc, 'xdisc data set')







