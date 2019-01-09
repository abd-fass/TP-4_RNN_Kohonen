# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:33:10 2019

@author: FMA
"""

import numpy as np
import scipy.io 
import matplotlib.pyplot as plt


def affiche_grille(w, x, title):
    
    plt.figure()
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.plot(x[0,:], x[1,:], 'bo')
    for b in range(np.shape(w)[0]):
        t1 = w[b,:,0]
        t2 = w[b,:,1]
        plt.plot(t1[:], t2[:], 'r*')
        plt.hold(True)
    for b in range(np.shape(w)[1]):
        t1 = w[:,b,0]
        t2 = w[:,b,1]
        plt.plot(t1[0], t2[0], 'r*')
        plt.hold(True)

    


data = scipy.io.loadmat('data.mat')
xdisc = data.get('xdisc')
xunif = data.get('xunif')

plt.figure()
plt.title('xdisc data representation')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.plot(xdisc[0,:], xdisc[1,:], 'o')
plt.show()

plt.figure()
plt.title('xunif data representation')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.plot(xunif[0,:], xunif[1,:], 'o')
plt.show()


"""2D Kohonen Map"""
K = 8
n = 2

nbiter = 10000

X = np.copy(xdisc)
length_x = np.shape(X)[1]

w = np.random.random((K,K,n))

wp = np.copy(w)
mu = np.zeros((1,nbiter))
sigma = np.zeros((1,nbiter))
pas_mu = (0.5 - 0.1)/(nbiter - 1)
pas_sigma = (3.0 - 0.1)/(nbiter - 1)
mu[0,0] = 0.5
sigma[0,0] = 3.0
for m in range(nbiter - 1):
    mu[0,m+1] = mu[0,m] - pas_mu
    sigma[0,m+1] = sigma[0,m] - pas_sigma



dist = np.ones((K,K))

for i in range(nbiter):
    
    coord_rand = np.random.randint(length_x)
    
    for x in range(K):
        for y in range(K):
            
            dist[x,y] =  np.sqrt((X[0,coord_rand] - wp[x,y,0])**2 + (X[1,coord_rand] - wp[x,y,1])**2)
    
    #coord_w_elu = np.where(dist == dist.min())
    #w_elu  = wp[coord_w_elu]
    
    min_dist = dist.min()
            
    for l in range(K):
        for c in range(K):
            if (dist[l,c] == min_dist):
                coord_w_elu_X = l
                coord_w_elu_Y = c
    
    for x in range(K):
        for y in range(K):
            
            A = np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2))
            B = A / (2*(sigma[0,i]**2))
            C = np.exp(-B)
            h = mu[0,i]*C
            #h = mu[0,i]*np.exp( np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2)) / (2*(sigma[0,i]**2)) )
            
            wp[x,y,0] = wp[x,y,0] + h*(X[0,coord_rand] - wp[x,y,0])
            wp[x,y,1] = wp[x,y,1] + h*(X[1,coord_rand] - wp[x,y,1])
    
    print('itération = ', i)
        
    
titre = 'Result Kohonen 2D xdisc'
affiche_grille(wp, X, titre) 





