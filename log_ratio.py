# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def log_ratio(img1, img2):
    """ Calcule le log ratio entre deux images.

    La fonction fait prend le log de la division des deux images, pixel par pixel.
    Les images doivent être de même taille.
    Les tableaux doivent continer des entiers representants les niveaux de gris.
    Les valeurs seront decalees de 1 afin d'eviter les valeurs a 0.

    :param img1: la première image à comparer
    :param img2: la deuxième image à comparer
    :type img1: numpy array a deux dimensions de int
    :type img2: numpy array a deux dimensions de int
    :return: retourne |log(X1 / X2)| où X1 appartient à img1 et X2 appartient à img2
             pour tous les pixels de img1 et img2
             retourne None si les dimensions sont irrecevables
    :rtype: numpy array a deux dimensions de int 
    """

    haut_img1 = len(img1)
    haut_img2 = len(img2)
    long_img1 = len(img1[0])
    long_img2 = len(img2[0])

    # Test des dimensions de l'images
    if haut_img1 == 0 or haut_img2 == 0 or haut_img1 != haut_img2 or \
        long_img1 == 0 or long_img2 == 0 or long_img1 != long_img2:
            return None

    # Initialisation du tableau qui contiendra le resultat du log ratio
    lr_img = np.zeros((haut_img1, long_img1))

    # Parcours des images
    for i in range(haut_img1):
        for j in range(long_img1):
            lr_img[i][j] = np.abs(np.log((img1[i][j] + 1) / (img2[i][j] + 1)))

    return lr_img


def main():
    """ Exemple d'utilisation du log ratio.

    La fonction donne deux exemples d'utilisation du log ratio: sur une image uniforme
    puis sur un degrade.
    """

    # parametres
    img_size = 255
    lambda_bruit = 10

    # creation de deux images unicolores
    img1 = np.array([[200 for i in range(img_size)] for j in range(img_size)])
    img2 = np.array([[200 for i in range(img_size)] for j in range(img_size)])

    # modification de la deuxieme image:
    # creation de deux lignes uniformes sur toute la longueur
    img2[img_size // 2] = [10 for i in range(img_size)]
    img2[(img_size // 2) + 1] = [10 for i in range(img_size)]

    # bruitage des images
    img1 = img1 + np.random.rayleigh(lambda_bruit, (img_size, img_size))
    img2 = img2 + np.random.rayleigh(lambda_bruit, (img_size, img_size))

    # log ratio
    lr_img = log_ratio(img1, img2)
    
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

    print("Nombre de pixels changés:         ", img_size * 2)
    print("Valeurs entre 0 inclus et 1 exclu:", q3)
    print("Valeurs entre 1 inclus et 2 exclu:", q2)
    print("Valeurs supérieures a 2 inclus:   ", q1)
    print()

    # affiche les images en niveaux de gris
    plt.subplot(131) # premiere image
    plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(132) # deuxieme image
    plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(133) # image des differences
    plt.imshow(lr_img, cmap = "gray", vmin = 1, vmax = 256)
    plt.show()


    # creation de deux images unicolores
    img1 = np.array([[200 for i in range(img_size)] for j in range(img_size)])
    img2 = np.array([[200 for i in range(img_size)] for j in range(img_size)])

    # modification de la deuxieme image:
    # creation de deux lignes uniformes sur toute la longueur
    img2[img_size // 2] = [10 for i in range(img_size)]
    img2[(img_size // 2) + 1] = [10 for i in range(img_size)]

    # bruitage multiplicatif des images
    img1 = img1 * np.random.rayleigh(lambda_bruit, (img_size, img_size))
    img2 = img2 * np.random.rayleigh(lambda_bruit, (img_size, img_size))

    # log ratio
    lr_img = log_ratio(img1, img2)
    
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

    print("Nombre de pixels changés:         ", img_size * 2)
    print("Valeurs entre 0 inclus et 1 exclu:", q3)
    print("Valeurs entre 1 inclus et 2 exclu:", q2)
    print("Valeurs supérieures a 2 inclus:   ", q1)
    print()

    # affiche les images en niveaux de gris
    plt.subplot(131) # premiere image
    plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(132) # deuxieme image
    plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(133) # image des differences
    plt.imshow(lr_img, cmap = "gray", vmin = 1, vmax = 256)
    plt.show()


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

    print("Nombre de pixels changés:         ", img_size * 2)
    print("Valeurs entre 0 inclus et 1 exclu:", q3)
    print("Valeurs entre 1 inclus et 2 exclu:", q2)
    print("Valeurs supérieures a 2 inclus:   ", q1)

    # affiche les images en niveaux de gris
    plt.subplot(131) # premiere image
    plt.imshow(img1, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(132) # deuxieme image
    plt.imshow(img2, cmap = "gray", vmin = 1, vmax = 256)
    plt.subplot(133) # image des differences
    plt.imshow(lr_img, cmap = "gray", vmin = 1, vmax = 256)
    plt.show()

if __name__ == '__main__':
    main()

