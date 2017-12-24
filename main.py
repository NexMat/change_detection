import numpy as np
import matplotlib.pyplot as plt
import util_test as utest
from lr_change_detection import lr_change_detection
from modelisation_detection import modelisation_detection

def optim_param():

    img_size = 256

    # creating images and changes
    img, mask, img_modif = utest.create_image(img_size, lambda_r=50)
    total_change, total_unchanged = utest.count_changes(mask)

    # log ratio method
    lr_detection_probas = lr_change_detection(img, img_modif)
    lr_detections_p  = [] # number of postive
    lr_detections_fp = [] # number of false positive

    lr_compromis_optim = 0
    lr_detection_optim = None

    # modelisation method
    mod_detection_probas = modelisation_detection(img, img_modif)
    mod_detections_p  = [] # number of postive
    mod_detections_fp = [] # number of false positive

    mod_compromis_optim = 0
    mod_detection_optim = None

    for i in range(1, 100):

        seuil_proba = i / 100

        lr_detection  = np.zeros((img_size, img_size))
        mod_detection = np.zeros((img_size, img_size))

        # treshold
        for j in range(img_size):
            for k in range(img_size):
                if lr_detection_probas[j][k] >= seuil_proba:
                    lr_detection[j][k]  = 255
                    mod_detection[j][k] = 255

        # computing the accuracy
        lr_positive, lr_false_positive = utest.count_detection(lr_detection, mask)
        lr_positive_ratio = lr_positive / total_change
        lr_false_positive_ratio = lr_false_positive / total_unchanged
        lr_detections_p.append(lr_positive_ratio)
        lr_detections_fp.append(lr_false_positive_ratio)

        mod_positive, mod_false_positive = utest.count_detection(mod_detection, mask)
        mod_positive_ratio = mod_positive / total_change
        mod_false_positive_ratio = mod_false_positive / total_unchanged
        mod_detections_p.append(mod_positive_ratio)
        mod_detections_fp.append(mod_false_positive_ratio)

        # calcul du compromis
        lr_compromis = lr_positive_ratio**2 + (1 - lr_false_positive_ratio)
        mod_compromis = mod_positive_ratio**2 + (1 - mod_false_positive_ratio)

        if lr_compromis >= lr_compromis_optim:
            lr_compromis_optim = lr_compromis
            lr_detection_optim = lr_detection

        if mod_compromis >= mod_compromis_optim:
            mod_compromis_optim = mod_compromis
            mod_detection_optim = mod_detection
        


    # displaying results 
    print(lr_compromis_optim)
    plt.title("Log ratio method")
    plt.xlabel('Fausses alarmes (%)')
    plt.ylabel('Bonne détection (%)')
    plt.plot(lr_detections_fp, lr_detections_p, '+b')
    plt.show()
    utest.img_show(img, img_modif, lr_detection_optim)

    print(mod_compromis_optim)
    plt.title("Modelisation method")
    plt.xlabel('Fausses alarmes (%)')
    plt.ylabel('Bonne détection (%)')
    plt.plot(mod_detections_fp, mod_detections_p, '+b')
    plt.show()
    utest.img_show(img, img_modif, mod_detection_optim)

if __name__ == '__main__':
    optim_param()
