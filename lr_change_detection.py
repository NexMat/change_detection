# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from log_ratio import log_ratio
from parcours_img import parcourir_image
from estim_dist import estimate_parameter

# TODO read_opt

def assign_proba_lr(lr_img):
    """ Assigne des probabilités a chaque pixel et a son voisinage
    a partir de l'image resultante du log ratio

    :param lr_img: l'image resultante de l'operation de log ratio
    :type  lr_img: numpy array a deux dimensions
    :return: retourne un tableau contenant un tableau de taille de
             l'image contenant la proba de chaque pixel d'etre un
             changement
    """

    voisinage = parcourir_image(lr_img)

    proba = np.zeros((len(lr_img), len(lr_img[0])))

    for i in range(len(lr_img)):
        for j in range(len(lr_img[0])):
            for lig in voisinage[i][j]:
                for val in lig:

                    if val == None:
                        pass

                    elif val >= 2:
                        proba[i][j] += 0.11

                    elif val > 1:
                        proba[i][j] += 0.05

    return proba
        


def test():

    # parametres
    img_size = 255
    lambda_bruit = 20

    # creation de deux images degrades
    img1 = np.array([[i for i in range(img_size)] for j in range(img_size)])
    img2 = np.array([[i for i in range(img_size)] for j in range(img_size)])

    # modification de la deuxieme image:
    # creation de deux lignes uniformes sur toute la longueur
    img2[img_size // 2] = [10 for i in range(img_size)]
    img2[(img_size // 2) + 1] = [10 for i in range(img_size)]

    # bruitage des images
    img1 = img1 + np.random.rayleigh(lambda_bruit, (img_size, img_size))
    img2 = img2 + np.random.rayleigh(lambda_bruit, (img_size, img_size))

    # log ratio
    lr_img = log_ratio(img1, img2)

    proba = assign_proba_lr(lr_img)

    q1, q2, q3 = 0, 0, 0
    # modification des valeurs pour l'affichage
    for i in range(img_size):
        for j in range(img_size):
            if lr_img[i][j] >= 2:
                lr_img[i][j] = 250
                q1 += 1
            elif lr_img[i][j] >= 1:
                lr_img[i][j] = 200
                q2 += 1
            elif lr_img[i][j] >= 0:
                q3 += 1
                lr_img[i][j] = 50

    #print("Nombre de pixels changés:         ", img_size * 2)
    #print("Valeurs entre 0 inclus et 1 exclu:", q3)
    #print("Valeurs entre 1 inclus et 2 exclu:", q2)
    #print("Valeurs supérieures a 2 inclus:   ", q1)

    valeurs = []
    for lgn in proba:
        #print(lgn)
        for val in lgn:
            if val != 0 and val not in valeurs:
                valeurs.append(val)

    valeurs = np.sort(valeurs)

    print("Probabilités assignées:")
    for val in valeurs:
        print(val)

    print("Pour quitter: q")
    while True:

        try:
            param_seuil = input("Entrez la valeur du seuil: ")
        except EOFError:
            print()
            break

        if param_seuil == 'q':
            break
        else:
            param_seuil = float(param_seuil)

        lr_img_smooth = np.zeros((img_size, img_size))

        for i in range(img_size):
            for j in range(img_size):
                if proba[i][j] >= param_seuil: # parametre
                    lr_img_smooth[i][j] = 250

        # affiche les images en niveaux de gris
        plt.subplot(141) # premiere image
        plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
        plt.subplot(142) # deuxieme image
        plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
        plt.subplot(143) # image des differences
        plt.imshow(lr_img, cmap = "gray", vmin = 1, vmax = 256)
        plt.subplot(144) # image des differences lissee
        plt.imshow(lr_img_smooth, cmap = "gray", vmin = 1, vmax = 256)
        plt.show()

def test_param():

    # parametres
    img_size = 255
    lambda_bruit = 50

    # creation de deux images degrades
    #img1 = np.array([[i for i in range(img_size)] for j in range(img_size)])
    #img2 = np.array([[i for i in range(img_size)] for j in range(img_size)])

    # creation de deux images a partir d'une distribution de rayleigh
    img1 = np.random.rayleigh(lambda_bruit, (img_size, img_size))
    img2 = np.random.rayleigh(lambda_bruit, (img_size, img_size))

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

    img_orig = np.copy(img2)

    # bruitage des images
    #img1 = img1 + np.random.rayleigh(lambda_bruit, (img_size, img_size))
    #img2 = img2 + np.random.rayleigh(lambda_bruit, (img_size, img_size))

    # log ratio
    lr_img = log_ratio(img1, img2)

    proba = assign_proba_lr(lr_img)

    q1, q2, q3 = 0, 0, 0
    # modification des valeurs pour l'affichage
    for i in range(img_size):
        for j in range(img_size):
            if lr_img[i][j] >= 2:
                lr_img[i][j] = 250
                q1 += 1
            elif lr_img[i][j] >= 1:
                lr_img[i][j] = 200
                q2 += 1
            elif lr_img[i][j] >= 0:
                q3 += 1
                lr_img[i][j] = 50

    #print("Nombre de pixels changés:         ", img_size * 2)
    #print("Valeurs entre 0 inclus et 1 exclu:", q3)
    #print("Valeurs entre 1 inclus et 2 exclu:", q2)
    #print("Valeurs supérieures a 2 inclus:   ", q1)

    #valeurs = []
    #for lgn in proba:
    #    #print(lgn)
    #    for val in lgn:
    #        if val != 0 and val not in valeurs:
    #            valeurs.append(val)

    #valeurs = np.sort(valeurs)

    positifs = []
    fausse_alarme = []

    seuil_optimal = 0
    compromis = 0
    lr_img_smooth_opti = None

    for i in range(100):

        param_seuil = i / 100

        lr_img_smooth = np.zeros((img_size, img_size))

        for j in range(img_size):
            for k in range(img_size):
                if proba[j][k] >= param_seuil: # parametre
                    lr_img_smooth[j][k] = 250

        count_positive, count_FA = count_detection(img_orig, lr_img_smooth)

        # Normalisation
        count_positive /= (img_size * 2)
        count_FA /= ((img_size**2) - (img_size * 2))

        positifs.append(count_positive)
        fausse_alarme.append(count_FA)

        compromis_tmp = count_positive + (1 - count_FA)

        if compromis_tmp >= compromis:
            compromis = compromis_tmp
            seuil_optimal = param_seuil
            lr_img_smooth_opti = lr_img_smooth.copy()


    plt.xlabel('Fausses alarmes (%)')
    plt.ylabel('Bonne détection (%)')
    plt.plot(fausse_alarme, positifs, '+b')
    plt.show()

    print(seuil_optimal)

    # affiche les images en niveaux de gris
    plt.subplot(141) # premiere image
    plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(142) # deuxieme image
    plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(143) # image des differences
    plt.imshow(lr_img, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(144) # image des differences lissee
    plt.imshow(lr_img_smooth_opti, cmap = "gray", vmin = 1, vmax = 256)
    plt.show()

def count_detection(img_orig, detection):
    count_FA = 0
    count_positive = 0

    for i in range(len(img_orig)):
        for j in range(len(img_orig[0])):
            if i == (len(img_orig) // 2) or i == (len(img_orig) // 2) + 1:
                if detection[i][j] == 250:
                    count_positive += 1

            elif detection[i][j] == 250:
                count_FA += 1
    
    return count_positive, count_FA



if __name__ == '__main__':
    test_param()
