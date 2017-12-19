# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from log_ratio import log_ratio
from parcours_img import parcourir_image
from parcours_img import index_voisins
from estim_dist import estimate_parameter


# TODO read_opt

threshold = 0.8
p_i = 0.2 # prior
p_v = 0.1 # observation

def assign_proba(img, lambda_est):

    global threshold
    global proba_assigned

    haut_img = len(img)
    long_img = len(img[0])

    voisinage = parcourir_image(img)

    proba = np.zeros((haut_img, long_img))

    for i in range(len(img)):
        for j in range(len(img[0])):

            index_vsn = index_voisins(i, j, haut_img, long_img)

            # p_v
            for index in index_vsn:
                if index != None and index != (i, j):
                    k, l = index
                    if is_superior(img[k][l], lambda_est, threshold):
                        proba[i][j] += p_v 


            # p_i
            if is_superior(img[i][j], lambda_est, threshold): #TODO adjust probability
                proba[i][j] += p_i

    return proba

def is_superior(pixel_value, dist_param, threshold):
    """
    threshold: percentage
    """

    cdf = lambda x : 1 - (np.exp(-(x**2) / (2 * (dist_param**2))))
    #cdf_inv = lambda x : dist_param * np.sqrt(-2 * np.log(1-x))

    #threshold_pixel_intensity = cdf_inv(threshold)
    #print("threshold:", threshold)
    #print("threshold_pixel_intensity:", threshold_pixel_intensity)

    perc = cdf(pixel_value)
    
    #print(perc, ">", threshold, "?")
    return perc >= threshold


def test():
    
    # parametres
    img_size = 100
    lambda_r = 80

    # creation de deux images a partir d'une distribution de rayleigh
    img1 = np.random.rayleigh(lambda_r, (img_size, img_size))
    img2 = np.random.rayleigh(lambda_r, (img_size, img_size))

    for i in range(img_size):
        for j in range(img_size):
            if img1[i][j] > 255:
                img1[i][j] = 255

            if img2[i][j] > 255:
                img2[i][j] = 255

    # modification de la deuxieme image:
    # creation de deux lignes uniformes sur toute la longueur
    img2[img_size // 2] = [250 for i in range(img_size)]
    img2[(img_size // 2) + 1] = [250 for i in range(img_size)]

    echantillon = np.array([lgn[-40:] for lgn in img2[-40:]]) # echantillon de taille 40x40 en bas à droite

    params = estimate_parameter(echantillon, "rayleigh")
    lambda_est = params[1]

    proba = assign_proba(img2, lambda_est)

    # valeurs des probabilités prises
    val_probas = {}
    for lgn in proba:
        for val in lgn:
            if val not in val_probas.keys():
                val_probas[val] = 1
            else:
                val_probas[val] += 1

    for key in sorted(val_probas.keys()): 
        print(key, val_probas[key])

    #for i in range(100):

    #param_seuil = i / 100
    param_seuil = 0.7

    # initialisation de la detection finale
    adv_detection = np.zeros((img_size, img_size))

    for j in range(img_size):
        for k in range(img_size):
            if proba[j][k] >= param_seuil:
                adv_detection[j][k] = 250

    # affiche les images en niveaux de gris
    plt.subplot(131) # premiere image
    plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(132) # deuxieme image
    plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(133) # image des differences
    plt.imshow(adv_detection, cmap = "gray", vmin = 1, vmax = 256)
    plt.show()


def plot_cumulative():
    
    size = 1000
    param = 80
    percentage_treshold = 0.6
    print(is_superior(150, param, percentage_treshold))
    data = np.random.rayleigh(80, size)

    values, base = np.histogram(data, bins=255) # evaluate the histogram
    cumulative = np.cumsum((values / size) * 100) # evaluate the cumulative (normalized)

    plt.subplot(121)
    plt.hist(data) # plot the histogram
    plt.subplot(122)
    plt.plot(cumulative) # plot the cumulative function
    
    plt.show()


if __name__ == '__main__':
    test()

