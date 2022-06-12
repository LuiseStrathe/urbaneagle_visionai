"""
These functions load the required raw data.
"""

#------------------------------------------------------------#

import numpy as np
import json
from keras.utils import load_img
from keras.utils import img_to_array
from matplotlib import pyplot as plt

#------------------------------------------------------------#


def import_image(path_raw_data, name_raw_data, raw_image_number):

    # load the image to get its shape
    orig_image = load_img(path_raw_data+name_raw_data+'.jpg')
    i_width, i_height = orig_image.size

    # convert to numpy array
    image = img_to_array(orig_image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # load the label-points for the picture from label studio
    with open(path_raw_data+name_raw_data+'.json') as json_file:
        json_load = json.load(json_file)

    # select data
    json_load = json_load[raw_image_number]["annotations"][0]["result"]
    labels_x = ([])
    labels_y = ([])
    for i in json_load:
            labels_x = np.append(labels_x, i["value"]["x"])
            labels_y = np.append(labels_y, i["value"]["y"])
    labels_unscaled = np.vstack((labels_y, labels_x)).transpose()

    # scale data points to image pixels
    reshaper = [[(i_height/100), 0], [0,(i_width/100)]]
    labels = np.mat(labels_unscaled) * np.mat(reshaper)
    print(f"There are {labels_unscaled[:,0].shape} labeled trees in the picture")

    # show image
    plt.imshow(orig_image)
    plt.show()
    plt.close()
    # check label distribution in histogram
    plt.hist((labels), bins=200)
    plt.show()
    print(f"The image size is {i_height}x{i_width} pixels.")
    print(f"The {len(labels_x)} data points are distributed along the respective axis as follows:")

    # add row for dedicated tile
    labels = labels.astype('int')
    labels = np.hstack((labels, 0*labels[:,0]))

    return image, labels, orig_image, labels_unscaled, i_width, i_height

