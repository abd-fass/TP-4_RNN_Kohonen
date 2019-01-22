# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:12:21 2019

@author: FMA
"""

import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import functions as f
import cv2


img = cv2.imread('FEMME.ppm')

cv2.namedWindow('FEMME', cv2.WINDOW_NORMAL)
cv2.imshow('FEMME', img)


x_data = np.zeros((3,(256*256)))

ind = 0
for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
        x_data[0,ind] = img[i,j,0]
        x_data[1,ind] = img[i,j,1]
        x_data[2,ind] = img[i,j,2]
        ind = ind + 1

""" Kohonen3D """

K = 8
nbiter = 10000
sigma_max = 3.0
sigma_min = 0.1
mu_max = 0.5
mu_min = 0.1

name_data = "FEMME"
nbr_affichage = 10

w = f.kohonen3d(x_data, K, mu_max, mu_min, sigma_max, sigma_min, nbiter, nbr_affichage, name_data)

""" Representation of the weigths result """

f.affiche_w_kohonen3D_final(w, K, name_data)

""" Coding """

img_code = f.Coding_Kohonen3D(w, K, img)

""" Decoding """

img_decode = f.Decoding_Kohonen3D(w, img_code)


plt.figure()
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Kohonen3D result image")
plt.imshow(cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB))
plt.show()

""" Adding noise perturbation """

#Matrix for the image code with noise
img_code_noise = np.zeros(np.shape(img_code), dtype = np.uint8())

for row in range(np.shape(img_code)[0]):
    for col in range(np.shape(img_code)[1]):
        for band in range(np.shape(img_code)[2]):
            
            #Value of noise (-1, 0 or 1)
            noise_value = np.random.randint(low = -1, high = 1)
        
            #Verify if the the coordinate of the neuron is not 0
            if ((noise_value == -1) and (img_code[row, col, band] == 0)):
                
                #if neuron coordinate is 0 don't apply the noise perturbation of -1 value to avoid overflow
                img_code_noise[row, col, band] = img_code[row, col, band] + ( (-1) * noise_value )
            
            #Verify if the the coordinate of the neuron is not 0    
            elif ((noise_value == +1) and (img_code[row, col, band] == K-1)):
                
                #if neuron coordinate is 7 don't apply the noise perturbation of +1 value to avoid overflow
                img_code_noise[row, col, band] = img_code[row, col, band] + ( (-1) * noise_value )
        
            else:
                
                img_code_noise[row, col, band] = img_code[row, col, band] + noise_value 

    print(round(((row/(np.shape(img_code)[0]-1))*100), 2), ' % Done (Code Noise)')


img_decode_noise = f.Decoding_Kohonen3D(w, img_code_noise)


plt.figure()
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Kohonen3D result image with noise")
plt.imshow(cv2.cvtColor(img_decode_noise, cv2.COLOR_BGR2RGB))
plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()
 
