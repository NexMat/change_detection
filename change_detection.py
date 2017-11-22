# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from log_ratio import log_ratio
from parcours_img import parcourir_image
from estim_dist import estimate_parameter


# TODO read_opt

def assign_proba(img):

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
