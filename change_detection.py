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
p_i = 0.1 # prior
p_v = 0.4 # observation

def assign_proba(img, lambda_est):

    global threshold
    global proba_assigned

    haut_img = len(img)
    long_img = len(img[0])

    voisinage = parcourir_image(img)

    proba = np.zeros((haut_img, long_img))

    for i in range(len(img)):
        for j in range(len(img[0])):

            # p_v
            for k,l in index_voisins(i, j, haut_img, long_img):
                if (i, j) != (k, l):
                    if is_superior(img[k][l], lambda_est, threshold):
                        proba[i][j] += p_v #TODO


            # p_i
            if is_superior(img[i][j], lambda_est, threshold):
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
    img2 = np.copy(img1)

    # modification de la deuxieme image:
    # creation de deux lignes uniformes sur toute la longueur
    img2[img_size // 2] = [10 for i in range(img_size)]
    img2[(img_size // 2) + 1] = [10 for i in range(img_size)]

    echantillon = np.array([lgn[-40:] for lgn in img2[-40:]]) # echantillon de taille 40x40 en bas à droite

    params = estimate_parameter(echantillon, "rayleigh")
    lambda_est = params[1]

    probs = assign_prob(img2, lambda_est)




if __name__ == '__main__':
    size = 1000
    param = 80
    percentage_treshold = 0.6
    print(is_superior(150, param, percentage_treshold))
    data = np.random.rayleigh(80, size)

    values, base = np.histogram(data, bins=255) # evaluate the histogram
    cumulative = np.cumsum((values / size) * 100) # evaluate the cumulative (normalized)

    plt.subplot(121)
    plt.hist(data) # plot the cumulative function
    plt.subplot(122)
    plt.plot(cumulative) # plot the survival function
    
    plt.show()

