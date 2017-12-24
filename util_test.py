import numpy as np
import matplotlib.pyplot as plt

def square_mask(img_size, thickness=10):
    """ crée un masque de changement carre creux d'une certaine epaisseur
    pixel = 0 si pas de changement
    pixel = 255 si changement
    """
    mask = np.zeros((img_size, img_size))

    for i in range(img_size // 4, 3 * (img_size // 4)):
        # verticales
        if i < (img_size // 4) + thickness or i > 3 * (img_size // 4) - thickness:
            for j in range(img_size // 4, 3 * (img_size // 4)):
                mask[i][j] = 255

        # horizontales
        else:
            for j in range(img_size // 4, 3 * (img_size // 4)):
                if j < (img_size // 4) + thickness or j > 3 * (img_size // 4) - thickness:
                    mask[i][j] = 255

    return mask

def apply_mask(img, mask):
    """ applique le masque de changement sur une image"""
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j] == 255:
                img[i][j] = 255

def create_image(img_size, change="square"):
    """ créer une image de test
    retourne l'image originale bruitée, le changement et l'image bruitée avec un changement
    """

    #TODO implementer d'autres changements

    lambda_r = 80
    
    # image originale
    img = np.random.rayleigh(lambda_r, (img_size, img_size))
    # image à modifier
    img_modif = np.random.rayleigh(lambda_r, (img_size, img_size))

    # valeurs au-dessus de l'intensite de pixel maximale
    for i in range(img_size):
        for j in range(img_size):
            if img1[i][j] > 255:
                img1[i][j] = 255

            if img2[i][j] > 255:
                img2[i][j] = 255

    # masque de changement
    mask = square_mask(img_size)
    # application du changement
    apply_mask(img_modif, mask)

    return img, changement, img_modif

def count_detection(detection, mask):
    """Compare la detection finale avec le mask d'origine
    sur les deux matrices:
        pixel = 0 si pas de changement
        pixel = 255 si changement
    return: le nombre de bonne détection, le nombre de faux positifs, le nombre total a detecter
    """

    positive = 0
    false_positive = 0
    total_change   = 0

    for i in range(len(detection)):
        for j in range(len(detection[0])):

            if detection[i][j] == 255:

                # positive
                if mask[i][j] == 255:
                    positive += 1

                # false positive
                elif mask[i][j] == 0:
                    false_positive += 1

            if mask[i][j] == 255:
                total_change += 1
    return positive, false_positive, total_change


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

if __name__ == '__main__':
    #plot_cumulative()

    mask = square_mask(255)
    plt.imshow(mask, cmap = "gray", vmin = 0, vmax = 255)
    plt.show()