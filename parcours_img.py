# librairies scientifiques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parcourir_image(img, rayon = 1):

    haut_img = len(img)
    long_img = len(img[0])

    voisinages = [[[[]] for j in range(long_img)] for i in range(haut_img)]

    for i in range(haut_img):
        for j in range(long_img):
            index = index_voisins(i, j, rayon, haut_img, long_img)
            voisins_tmp = [[None for i_ in range(rayon * 2 + 1)] for j_ in range(rayon * 2 + 1)]
            for k in range(rayon * 2 + 1):
                for l in range(rayon * 2 + 1):
                    if index[k * (rayon * 2 + 1) + l] != None:
                        voisins_tmp[k][l] = img[index[k * (rayon * 2 + 1) + l][0]][index[k * (rayon * 2 + 1) + l][1]]

                    else:
                        voisins_tmp[k][l] = None


            voisinages[i][j] = voisins_tmp

    return voisinages

def is_in_array(i, j, haut_img, long_img):
    return i >= 0 and j >= 0 and i < haut_img and j < long_img

def index_voisins(i, j, rayon = 1, haut_img, long_img):
    voisins = []
    for shift_i in range(-rayon, rayon + 1):
        for shift_j in range(-rayon, rayon + 1):
            #print("voisins", shift_i + i, shift_j + j)
            if is_in_array(i + shift_i, j + shift_j, haut_img, long_img):
                voisins.append((i + shift_i, j + shift_j))
            else:
                voisins.append(None)
    return voisins

if __name__ == '__main__':
    
    img_size = 256

    img = np.array([[i for i in range(img_size)] for j in range(img_size)])

    #plt.imshow(img, cmap = "gray", vmin = 1, vmax = 256)
    #plt.show()

    voisinages = parcourir_image(img, 1)

    for i in range(3):
        for j in range(3):
            print("voisins de", i * 127, j * 127, ":")
            for l in voisinages[i * 127][j * 127]:
                print(l)
            print()
            #plt.imshow(voisinages[i * 127][j * 127], cmap = "gray", vmin = 1, vmax = 256)
            #plt.show()
