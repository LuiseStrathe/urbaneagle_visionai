"""
ABOUT
This script is used to segment a model into water, green, building and else areas.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
import matplotlib.pyplot as plt
import numpy as np

name = "Dresden_01"
#image_path = f'../data/raw/images_unlabeled/{name}.jpg'
image_path = f'../data/raw/segmentation/{name}/{name}.jpg'
report_path = '../reports/'
report_path = report_path + name + '/'

## in pixel:
tile_size = 512

# model specification
path_model = '../models/'
model_name = "seg_512.h5"
path_model = path_model + model_name
threshold = 0.1


# --------------------SETUP--------------------------#

from tqdm import tqdm

from src.data.seg_loader \
    import *


# ----------------DATA LOADING-----------------------#
print("\n--------------------------------------------"
      f"\nLoading Image for {name}...")

image, tiles, dims = \
    pred_load(
        image_path, tile_size, name)


# ----------------------IDENTIFICATION--------------#
print("\n--------------------------------------------"
      "\nStart prediction for segmentation...")

# identify tree-tiles
model = tf.keras.models.load_model(path_model)

probabilities = ([])
for pair in tqdm(range(len(tiles) // 2)):
    print(f"Predicting tile {pair*2}...")
    prob = model.predict(tiles[pair*2:pair*2+1, :, :, :])
    probabilities.append(prob)
    pass
probabilities = np.array(probabilities)
np.save(report_path + 'seg_pred.npy', probabilities)

def show_pred(image, probabilities, dims, report_path, name, tile_size):

    plt.imshow(image)
    plt.show()
    plt.imsave(f'{report_path}/{name}.jpg', image)
    plt.close()

    # concat tiles of prediction
    # dims: [num_tiles, num_ver, num_hor]
    segmentation = probabilities[:, :, :, :3].reshape(dims[1] * tile_size, dims[2] * tile_size, 3)

    plt.imshow(segmentation.squeeze())
    plt.title("Predicted segmentation")
    plt.show()
    plt.imsave(report_path + 'seg_pred.png')
    plt.close()

show_pred(image, probabilities, dims, report_path, name, tile_size)


# ----------------------END--------------------------#
print("\n---------------------------------------------"
      "\nPrediction of segments is finished!")
