# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def estimate_parameter(data, dist_name = "rayleigh"):
    """ Estime le parametre d'une distribution
    
    Estime le parametre de la distribution dont le nom est donne en
    argument suivant les donnees.

    :param data: les donnees a utiliser
    :param dist_name: le nom de la distribution a faire correspondre
    :type data: un numpy array de taille quelconque
    :type dist_name: un string
    :return: les parametres (taille et centre notamment)
    :rtype: tuple de float
    """

    dist = getattr(stats, dist_name)
    param = dist.fit(data)

    return param

def main():
    """
    """
    param = 20
    data_size = 1000

    data = np.random.rayleigh(param, data_size) + np.array([5 for i in range(data_size)])

    param_estime = estimate_parameter(data, "rayleigh")

    print(param_estime)

if __name__ == '__main__':
    main()

