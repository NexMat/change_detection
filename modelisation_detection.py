# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util_test as utest

from log_ratio import log_ratio
from estim_dist import estimate_parameter

threshold = 0.8

def assign_proba_mod(img, lambda_est, proba_x, proba_v, nbh_size):
    """
    proba_x and proba_v are between 0 and 1
    """

    global threshold

    h, w = img.shape

    proba = np.zeros((h, w))

    for i in range(h):
        for j in range(w):

            index_vsn = utest.index_voisins(i, j, (h, w), rayon = nbh_size)
            nb_vsn = len(index_vsn) - 1 # On calcule dynamiquement le nombre de voisins (notamment pour les bords)

            # p_v
            for index in index_vsn:
                if index != None and index != (i, j):
                    k, l = index
                    if is_superior(img[k][l], lambda_est, threshold):
                        proba[i][j] += (proba_v / nb_vsn) # On pondère en fonction du nombre de voisins


            # p_i
            if is_superior(img[i][j], lambda_est, threshold):
                proba[i][j] += proba_x

    return proba / 2

def is_superior(pixel_value, dist_param, threshold):
    """
    threshold: percentage
    """

    cdf = lambda x : 1 - (np.exp(-(x**2) / (2 * (dist_param**2))))
    perc = cdf(pixel_value)
    
    return perc >= threshold

def test():
    # parametres
    img_size = 100

    # creation d'images et de changements
    img, mask, img_modif = utest.create_image(img_size)


def old_test():
    
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

def modelisation_detection(img_orig, img_modif, proba_x, proba_v, nbh_size):

    """ computes the change detection with the log ratio
    return a black and white matrix indicating changes in white
    """

    # estimate the parameter
    echantillon = np.array([lgn[-40:] for lgn in img_modif[-40:]]) # echantillon de taille 40x40 en bas à droite
    loc, lambda_est = estimate_parameter(echantillon, "rayleigh")

    # assign to each pixel a probability
    proba_mod = assign_proba_mod(img_modif, lambda_est, proba_x, proba_v, nbh_size)

    return proba_mod

if __name__ == '__main__':
    test()

