# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:29:28 2019

@author: FMA
"""

import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
from random import shuffle
from random import randint
import cv2

def affiche_grille(w, x, title):
    
    fig = plt.figure()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.plot(x[0,:], x[1,:], 'b*')
    compt = 1
    for b in range(np.shape(w)[0]):
        t1 = w[b,:,0]
        t2 = w[b,:,1]
        for i in range(len(t1)):
            plt.text(t1[i], t2[i], 'N' + str(compt),  color='red', bbox=dict(facecolor='white', alpha=0.9))
            plt.plot(t1[i], t2[i], 'ro', markersize=30)
            plt.hold(True)
            compt = compt + 1
    
    plt.title(title + ' Neurones = ' + str(compt-1))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.pause(2)
    plt.close(fig)

    

def affiche_grille_final(w, x, title):
    
    plt.figure()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    plt.plot(x[0,:], x[1,:], 'b*')
    compt = 1
    for b in range(np.shape(w)[0]):
        t1 = w[b,:,0]
        t2 = w[b,:,1]
        for i in range(len(t1)):
            plt.text(t1[i], t2[i], 'N' + str(compt),  color='red', bbox=dict(facecolor='white', alpha=0.9))
            plt.plot(t1[i], t2[i], 'ro', markersize=30)
            plt.hold(True)
            compt = compt + 1

    plt.title(title + ' Neurones = ' + str(compt-1))
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

def kohonen2d(x, K, mu_max, mu_min, sigma_max, sigma_min, nbiter, nbr_affichage, name_data):
    
    #Dimension Map
    n = 2
    
    #the number of th plot display for the evolution od the neurons
    affichage_iter = np.zeros((nbr_affichage+1))
    pas = int(np.floor(nbiter/nbr_affichage))
    pred_value = 0
    for l in range(nbr_affichage):
        affichage_iter[l+1] = pred_value + pas
        pred_value = pred_value + pas
        
    compt = 0
    
    X = np.copy(x)
    length_x = np.shape(X)[1]
    
    #Random w
    w = np.random.random((K,K,n))
    
    wp = np.copy(w)
    
    #create mu and sigma vectors
    mu = np.zeros((1,nbiter))
    sigma = np.zeros((1,nbiter))
    for m in range(nbiter):
        mu[0,m] = mu_max + ( m / (nbiter - 1) ) * ( mu_min - mu_max)
        sigma[0,m] = sigma_max + ( m / (nbiter - 1) ) * ( sigma_min - sigma_max)
    
    #distance matrix
    dist = np.ones((K,K))
    
    
    for i in range(nbiter):
        
        #random sample coordinate
        coord_rand = np.random.randint(length_x)
        
        #Calculate the distance between all the neurons and the sample
        for x in range(K):
            for y in range(K):
                
                dist[x,y] =  np.sqrt((X[0,coord_rand] - wp[x,y,0])**2 + (X[1,coord_rand] - wp[x,y,1])**2)
        
        #Get min distance to get the winning neuron and it's coordinate
            
        coord_w_elu_X, coord_w_elu_Y = np.where(dist == dist.min())
        coord_w_elu_X = int(coord_w_elu_X[0])
        coord_w_elu_Y = int(coord_w_elu_Y[0])
        
        #Calculate the new location for all neurons
        for x in range(K):
            for y in range(K):
                
                A = np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2))
                B = A / (2*(sigma[0,i]**2))
                C = np.exp(-B)
                h = mu[0,i]*C
                #h = mu[0,i]*np.exp( np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2)) / (2*(sigma[0,i]**2)) )
                
                wp[x,y,0] = wp[x,y,0] + h*(X[0,coord_rand] - wp[x,y,0])
                wp[x,y,1] = wp[x,y,1] + h*(X[1,coord_rand] - wp[x,y,1])
        
        print(round(((i/(nbiter-1))*100), 2), ' % Done (Kohonen2D)')
        
        #Display the result for the chosen iteration
        if((i) == int(affichage_iter[compt])):
            
            titre = 'Résultat Kohonen 2D ' + name_data +  ' itération = ' + str(i) 
            affiche_grille(wp, X, titre) 
            compt = compt + 1
        
    return wp

def clas_Kohonen2D(x, K, w): 
    
    length_x = len(x[0,:])
    #Clas vector that contain the classification of each sample
    clas = np.zeros((1,length_x))

    #matrix distance that contain the distance between sample's and neuron's
    dist_clas = np.zeros((K,K))

    #matrix that contain the number of each neuron to allow identification of each classification     
    n_w = np.zeros((K,K))
    ind = 1
    for rw in range(K):
        for cl in range(K):
                
            n_w[K-1-rw, cl] = ind
            ind = ind + 1

    #loop to calculate the classification
    for i in range(length_x):
    
        #Calculate the distance between sample i and all the neurons
        for rw in range(K):
            for cl in range(K):
            
                dist_clas[rw,cl] = np.sqrt( ( (x[0,i] - w[rw,cl,0])**2 ) + ( (x[1,i] - w[rw,cl,1])**2 ) )
    
        #Get the min distance and it's coordinates
        coord_X_clas, coord_Y_clas = np.where( dist_clas == dist_clas.min() )
        coord_X_clas = int(coord_X_clas)
        coord_Y_clas = int(coord_Y_clas)
    
        #apply the classification
        clas[0,i] = n_w[coord_X_clas,coord_Y_clas] 
        
    return clas


def kohonen3d(x, K, mu_max, mu_min, sigma_max, sigma_min, nbiter, nbr_affichage, name_data):
    
    #the number of th plot display for the evolution od the neurons
    affichage_iter = np.zeros((nbr_affichage+1))
    pas = int(np.floor(nbiter/nbr_affichage))
    pred_value = 0
    for l in range(nbr_affichage):
        affichage_iter[l+1] = pred_value + pas
        pred_value = pred_value + pas
        
    compt = 0
    
    X = np.copy(x)
    length_x = np.shape(X)[1]
        
    #Random w
    w_x = np.random.random((K,K,K))
    w_y = np.random.random((K,K,K))
    w_z = np.random.random((K,K,K))
    
    wp_x = np.copy(w_x)
    wp_y = np.copy(w_y)
    wp_z = np.copy(w_z)
    
    wp = [wp_x, wp_y, wp_z] 
        
    #create mu and sigma vectors
    mu = np.zeros((1,nbiter))
    sigma = np.zeros((1,nbiter))
    for m in range(nbiter):
        mu[0,m] = mu_max + ( m / (nbiter - 1) ) * ( mu_min - mu_max)
        sigma[0,m] = sigma_max + ( m / (nbiter - 1) ) * ( sigma_min - sigma_max)
        
    #distance matrix
    dist = np.ones((K,K,K))
        
        
    for i in range(nbiter):
            
        #random sample coordinate
        coord_rand = np.random.randint(length_x)
            
        #Calculate the distance between all the neurons and the sample
        for x in range(K):
            for y in range(K):
                for z in range(K):
                    
                    dist[x,y,z] =  np.sqrt((X[0,coord_rand] - wp[0][x,y,z])**2 + (X[1,coord_rand] - wp[1][x,y,z])**2 + (X[2,coord_rand] - wp[2][x,y,z])**2)
            
        #Get min distance to get the winning neuron and it's coordinate
            
        coord_w_elu_X, coord_w_elu_Y, coord_w_elu_Z = np.where(dist == dist.min())
        coord_w_elu_X = int(coord_w_elu_X[0])
        coord_w_elu_Y = int(coord_w_elu_Y[0])
        coord_w_elu_Z = int(coord_w_elu_Z[0])
            
        #Calculate the new location for all neurons
        for x in range(K):
            for y in range(K):
                for z in range(K):
                    
                    A = np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2 + (z - coord_w_elu_Z)**2))
                    B = A / (2*(sigma[0,i]**2))
                    C = np.exp(-B)
                    h = mu[0,i]*C
                    #h = mu[0,i]*np.exp( np.sqrt(((x - coord_w_elu_X)**2 + (y - coord_w_elu_Y)**2)) / (2*(sigma[0,i]**2)) )
                    
                    wp[0][x,y,z] = wp[0][x,y,z] + h*(X[0,coord_rand] - wp[0][x,y,z])
                    wp[1][x,y,z] = wp[1][x,y,z] + h*(X[1,coord_rand] - wp[1][x,y,z])
                    wp[2][x,y,z] = wp[2][x,y,z] + h*(X[2,coord_rand] - wp[2][x,y,z])
        
        print(round(((i/(nbiter-1))*100), 2), ' % Done (Kohonen3D)')
        
        #Display the result for the chosen iteration
        if((i) == int(affichage_iter[compt])):
            
            affiche_w_kohonen3D(wp, K, name_data, i) 
            compt = compt + 1
        
    return wp

def Coding_Kohonen3D(w, K, img):
    
    #Matrix that contain all the distance value between the pixel i and all the neurons
    dist_code = np.ones((K,K,K))
    
    #Image code with neurons cooridnates
    img_code = np.zeros((np.shape(img)), dtype = np.uint8())
    
    for l in range(np.shape(img)[0]):
        for c in range(np.shape(img)[1]):
            
            #Get the 3 values of the pixel i
            pixel_x = img[l,c,0]
            pixel_y = img[l,c,1]
            pixel_z = img[l,c,2]
            
            #Calculate the distance between the pixel i and all the neurons 
            for x in range(K):
                for y in range(K):
                    for z in range(K):
                        
                        dist_code[x,y,z] = np.sqrt((pixel_x - w[0][x,y,z])**2 + (pixel_y - w[1][x,y,z])**2 + (pixel_z - w[2][x,y,z])**2)
            
            #Get the minimum distance to code the pixel i with the cooridnate of the neuron with distance = min and it's coordinate
            
            coord_w_code_X, coord_w_code_Y, coord_w_code_Z = np.where(dist_code == dist_code.min())
            coord_w_code_X = int(coord_w_code_X)
            coord_w_code_Y = int(coord_w_code_Y)
            coord_w_code_Z = int(coord_w_code_Z)
            
            #Affect the neuron coordinate's to the pixel
            img_code[l,c,0] = coord_w_code_X
            img_code[l,c,1] = coord_w_code_Y
            img_code[l,c,2] = coord_w_code_Z
            
        print(round(((l/(np.shape(img_code)[0]-1))*100), 2), ' % Done (Coding)')
        
    return img_code

def Decoding_Kohonen3D(w, img_code):
    
    #Image decode
    img_decode = np.zeros((np.shape(img_code)), dtype = np.uint8())
    
    
    for ligne in range(np.shape(img_code)[0]):
        for colonne in range(np.shape(img_code)[1]):
            
            #Replace the neuron cordiante with his 3 values (w)        
            img_decode[ligne, colonne, 0] = np.uint8(w[0][img_code[ligne,colonne,0],img_code[ligne,colonne,1],img_code[ligne,colonne,2]])
            img_decode[ligne, colonne, 1] = np.uint8(w[1][img_code[ligne,colonne,0],img_code[ligne,colonne,1],img_code[ligne,colonne,2]])
            img_decode[ligne, colonne, 2] = np.uint8(w[2][img_code[ligne,colonne,0],img_code[ligne,colonne,1],img_code[ligne,colonne,2]])
            
        print(round(((ligne/(np.shape(img_decode)[0]-1))*100), 2), ' % Done (Decoding)')
    
    return img_decode

def affiche_w_kohonen3D(w, K, name_data, niter):
    
    #Representation of the weigths result
    n = 3
    img_w = np.zeros(((K*K*K),(K*K*K),n))
    
    wp_x = np.copy(w[0])
    wp_y = np.copy(w[1])
    wp_z = np.copy(w[2])
    
    wpx_display = np.zeros((K,K*K))
    wpy_display = np.zeros((K,K*K))
    wpz_display = np.zeros((K,K*K))
    
    compt = 0
    for l in range(K):
        for row in range(K):
            for col in range(K):
                wpx_display[l,compt] = wp_x[row, col, l]
                wpy_display[l,compt] = wp_y[row, col, l]
                wpz_display[l,compt] = wp_z[row, col, l]
                compt = compt + 1
        compt = 0
    
    ind_k = 0
    compt = 0
    for col in range(np.shape(img_w)[1]):
        
            img_w[:,col,0] = wpx_display[ind_k,compt]
            img_w[:,col,1] = wpy_display[ind_k,compt]
            img_w[:,col,2] = wpz_display[ind_k,compt]
            
            if (compt >= ((K*K)-1)):
                compt = 0
                ind_k = ind_k + 1
            elif (compt < ((K*K)-1)):
                compt = compt + 1
        
    img_w = np.uint8(img_w)
    
    
    fig = plt.figure()
    plt.title("Representation of the weights result iter=" + str(niter) +  " (" + name_data  +")")
    plt.xlabel("w")
    plt.imshow(cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.pause(2)
    plt.close(fig)
    
def affiche_w_kohonen3D_final(w, K, name_data):
    
    #Representation of the weigths result
    n = 3
    img_w = np.zeros(((K*K*K),(K*K*K),n))
    
    wp_x = np.copy(w[0])
    wp_y = np.copy(w[1])
    wp_z = np.copy(w[2])
    
    wpx_display = np.zeros((K,K*K))
    wpy_display = np.zeros((K,K*K))
    wpz_display = np.zeros((K,K*K))
    
    compt = 0
    for l in range(K):
        for row in range(K):
            for col in range(K):
                wpx_display[l,compt] = wp_x[row, col, l]
                wpy_display[l,compt] = wp_y[row, col, l]
                wpz_display[l,compt] = wp_z[row, col, l]
                compt = compt + 1
        compt = 0
    
    ind_k = 0
    compt = 0
    for col in range(np.shape(img_w)[1]):
        
            img_w[:,col,0] = wpx_display[ind_k,compt]
            img_w[:,col,1] = wpy_display[ind_k,compt]
            img_w[:,col,2] = wpz_display[ind_k,compt]
            
            if (compt >= ((K*K)-1)):
                compt = 0
                ind_k = ind_k + 1
            elif (compt < ((K*K)-1)):
                compt = compt + 1
        
    img_w = np.uint8(img_w)
    
    
    plt.figure()
    plt.title("Final representation of the weights result (" + name_data  +")")
    plt.xlabel("w")
    plt.imshow(cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB))
    plt.show()

def affiche_clas_kohonen2D(x, clas, w, title):
    
    # Sort colors by hue, saturation, value and name.
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    random.shuffle(sorted_names)
    
    
#    coul = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
#            'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
#            'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
#            'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
    
    list_indice = []
    
    plt.figure()
    plt.xlabel('Dimension (1)')
    plt.ylabel('Dimension (2)')
    
    #Display data sample waith classification
    for i in range(int(np.amax(clas)+1)):
        for j in range(len(clas[0,:])):
        
            if (clas[0,j] == i):
                list_indice.append(j)
            
        list_indice = tuple(list_indice)
        plt.scatter(x[0,list_indice],x[1,list_indice], color = sorted_names[i])
        #plt.hold(True)
        list_indice = []
    
    #Display the neurons
    compt = 1
    for b in range(np.shape(w)[0]):
        t1 = w[b,:,0]
        t2 = w[b,:,1]
        for i in range(len(t1)):
            plt.text(t1[i], t2[i], 'N' + str(compt),  color='red', bbox=dict(facecolor='white', alpha=0.9))
            plt.plot(t1[i], t2[i], 'ro', markersize=30)
            plt.hold(True)
            compt = compt + 1
    
    plt.title('Kohonen2D data classification '+ title)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()