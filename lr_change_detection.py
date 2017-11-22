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

if __name__ == '__main__':
    test()
