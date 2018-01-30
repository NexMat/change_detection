import numpy as np
import matplotlib.pyplot as plt
import cv2
import util_test as utest
from lr_change_detection import lr_change_detection
from modelisation_detection import modelisation_detection


def optim_param():
    """La fonction compare les deux méthodes de détection
    Elle cherchera à optimiser les paramètres de chacunes.
    """

    img_size = 256

    # Creating images and changes
    img, mask, img_modif = utest.create_image(img_size, lambda_r=90)
    total_change, total_unchanged = utest.count_changes(mask)
    print('[+] Création des images')

    # Final result variables
    lr_compromis_optim = 0
    lr_detection_optim = None

    mod_compromis_optim = 0
    mod_detection_optim = None

    lr_detections_p  = [] # number of postive
    lr_detections_fp = [] # number of false positive

    mod_detections_p  = [] # number of postive
    mod_detections_fp = [] # number of false positive

    lr_param_optim  = 0, 0, 0
    mod_param_optim = 0, 0, 0

    print('[+] Calcul des paramètres')
    # Variation of probabilites assigned
    for proba_x_ in range(0, 100, 10):
        for proba_v_ in range(0, 100, 10):

            nbh_size = 2

            proba_x = proba_x_ / 100
            proba_v = proba_v_ / 100

            # Log ratio method
            lr_detection_probas = lr_change_detection(img, img_modif, proba_x, proba_v, nbh_size)
            # Modelisation method
            mod_detection_probas = modelisation_detection(img, img_modif, proba_x, proba_v, nbh_size)
        
            # Variation of threshold
            for seuil_proba_ in range(0, 100, 10):

                if seuil_proba_ == 0:
                    numero_iter = int(proba_x_ + proba_v_ / 10) + 1
                    print('[+] Iteration:', repr(numero_iter).rjust(2), '/ 100')

                seuil_proba = seuil_proba_ / 100

                lr_detection  = np.zeros((img_size, img_size))
                mod_detection = np.zeros((img_size, img_size))

                # Treshold
                lr_detection[lr_detection_probas >= seuil_proba] = 255
                mod_detection[mod_detection_probas >= seuil_proba] = 255

                # Computing the accuracy
                lr_positive, lr_false_positive = utest.count_detection(lr_detection, mask)
                lr_positive_ratio = lr_positive / total_change
                lr_false_positive_ratio = lr_false_positive / total_unchanged
                lr_detections_p.append(lr_positive_ratio)
                lr_detections_fp.append(lr_false_positive_ratio)
                #lr_detections_p[(proba_x, proba_v, seuil_proba)] = lr_positive_ratio
                #lr_detections_fp[(proba_x, proba_v, seuil_proba)] = lr_false_positive_ratio

                mod_positive, mod_false_positive = utest.count_detection(mod_detection, mask)
                mod_positive_ratio = mod_positive / total_change
                mod_false_positive_ratio = mod_false_positive / total_unchanged
                mod_detections_p.append(mod_positive_ratio)
                mod_detections_fp.append(mod_false_positive_ratio)
                #mod_detections_p[(proba_x, proba_v, seuil_proba)] = mod_positive_ratio
                #mod_detections_fp[(proba_x, proba_v, seuil_proba)] = mod_false_positive_ratio

                # Calcul du compromis # TODO parametriser les coefficients: + ou - pénaliser BD ou FA
                lr_compromis  = lr_positive_ratio  + (1 - lr_false_positive_ratio)
                mod_compromis = mod_positive_ratio + (1 - mod_false_positive_ratio)


                if lr_compromis > lr_compromis_optim:
                    lr_compromis_optim = lr_compromis
                    lr_detection_optim = lr_detection
                    lr_param_optim = proba_x, proba_v, seuil_proba

                    print('[!] Méthode LR')
                    print('[!]     Meilleurs paramètres obtenus:')
                    print('[!]         Proba x:', proba_x)
                    print('[!]         Proba v:', proba_v)
                    print('[!]         Seuil:  ', seuil_proba)
                    print('[!]     Compromis calculé:', lr_compromis / 2)
                    print('[!]')

                if mod_compromis > mod_compromis_optim:
                    mod_compromis_optim = mod_compromis
                    mod_detection_optim = mod_detection
                    mod_param_optim = proba_x, proba_v, seuil_proba

                    print('[!] Méthode mod')
                    print('[!]     Meilleurs paramètres obtenus:')
                    print('[!]         Proba x:', proba_x)
                    print('[!]         Proba v:', proba_v)
                    print('[!]         Seuil:  ', seuil_proba)
                    print('[!]     Compromis calculé:', mod_compromis / 2)
                    print('[!]')

    # Displaying results 
    print('[+] Log ratio compromis:', lr_compromis_optim / 2)
    print('[+] Paramètres optimums:',  lr_param_optim)
    plt.title("Log ratio method")
    plt.xlabel('Fausses alarmes (%)')
    plt.ylabel('Bonne détection (%)')
    plt.plot(lr_detections_fp, lr_detections_p, '+b')
    plt.show()
    utest.img_show(img, img_modif, lr_detection_optim)

    print("[+] Modelisation compromis:", mod_compromis_optim / 2)
    print('[+] Paramètres optimums:',  mod_param_optim)
    plt.title("Modelisation method")
    plt.xlabel('Fausses alarmes (%)')
    plt.ylabel('Bonne détection (%)')
    plt.plot(mod_detections_fp, mod_detections_p, '+b')
    plt.show()
    utest.img_show(img, img_modif, mod_detection_optim)

def detection_image():

    img = cv2.imread("L2_crop.png", cv2.IMREAD_GRAYSCALE)
    img_modif = cv2.imread("L3_inv_crop.png", cv2.IMREAD_GRAYSCALE)

    img = np.array(img * 255, dtype=np.int32)
    img_modif = np.array(img_modif * 255, dtype=np.int32)

    # Log ratio method
    lr_detection_probas = lr_change_detection(img, img_modif, 0, 0.7, 2)
    print('[+] Methode LR terminée')
    
    # Modelisation method
    mod_detection_probas = modelisation_detection(img, img_modif, 0.3, 0.9, 2)
    print('[+] Methode mod terminée')

    lr_detection  = np.zeros((img_size, img_size))
    mod_detection = np.zeros((img_size, img_size))
    
    print('[+] Seuillage')
    # Treshold
    lr_detection[lr_detection_probas >= 0.1] = 255
    mod_detection[mod_detection_probas >= 0.4] = 255
    
    utest.img_show(img, img_modif, lr_detection)

    utest.img_show(img, img_modif, mod_detection)

if __name__ == '__main__':
    #optim_param()
    detection_image()
