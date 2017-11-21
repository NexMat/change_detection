# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():


    img_size = 50

    img_rayleigh = np.random.rayleigh(200, (img_size, img_size))
    img_weibull  = np.random.weibull (200, (img_size, img_size))

    print(img_rayleigh)

    # affiche les images en niveaux de gris
    #plt.subplot(121) # premiere image
    plt.imshow(img_rayleigh, cmap = "gray", vmin = 0, vmax = 255)
    plt.show()


if __name__ == '__main__':
    main()
