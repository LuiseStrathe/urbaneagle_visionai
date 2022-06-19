"""
ABOUT
This script is used to segment a model into water, green, building and else areas.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
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
import numpy as np
import matplotlib.pyplot as plt

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
#for pair in tqdm(range(len(tiles) // 2)):
#    print(f"Predicting tile {pair*2} & {pair*2+1} of {len(tiles)}...")
#    prob = model.predict(tiles[(pair*2):(pair*2+2), :, :, :])
#    if pair == 0:
#        probabilities = prob
#    else: probabilities = np.concatenate((probabilities, prob), axis=0)

for tile in tqdm(range(len(tiles))):
    print(f"predicting tile {tile}")
    prob = model.predict(np.array([tiles[tile, :, :, :]]))
    if tile == 0:
        probabilities = prob
    else: probabilities = np.concatenate((probabilities, prob), axis=0)


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

    plt.imshow(segmentation)
    plt.title("Predicted segmentation")
    plt.show()
    plt.close()

show_pred(image, probabilities, dims, report_path, name, tile_size)


# ----------------------END--------------------------#
print("\n---------------------------------------------"
      "\nPrediction of segments is finished!")
