"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
name = "Dresden_06"
#image_path = f'../data/raw/images_unlabeled/{name}.jpg'
image_path = f'../data/raw/tree_detection/{name[:-3]}/{name}.jpg'
report_path = '../reports/'
report_path = report_path + name + '/'

## in pixel:
tile_size = 25
## in pixel per side, expands small tile to large tile:
border = 15

# model specification
path_model_ide = '../models/'
model_name_ide = "ide_55_th97"
path_model_ide = path_model_ide + model_name_ide + '/'
threshold_ide = 0.1

path_model_pos = '../models/'
model_name_pos = "pos"
path_model_pos = path_model_pos + model_name_pos


# --------------------SETUP--------------------------#

from src.data.tree_loader \
    import *
from src.models.identification_model \
    import *
from src.visualization.tree_vizualizations \
    import *


# ----------------DATA LOADING-----------------------#
print("\n--------------------------------------------"
      f"\nLoading Image for {name}...")

# load image
image, tiles_large, tiles, tile_info = \
    image_load_tiles(
        image_path, tile_size, border, name, report_path)


# ----------------------IDENTIFICATION--------------#
print("\n--------------------------------------------"
      "\nStart prediction for identification...")

# identify tree-tiles
model_ide = tf.keras.models.load_model(path_model_ide)
probabilities_ide = model_ide.predict(tiles_large)
tile_info[:, 3] = probabilities_ide[:, 0]
predictions_ide = np.array([1 if x >= threshold_ide else 0 for x in probabilities_ide])
show_pred_tiles(predictions_ide, tile_info, path_model=report_path, name_raw_data=name)


# ------------------POSITIONING------------------#
print("\n--------------------------------------------"
      "\nStart prediction for positioning of tre pixels...")

# create tile set with trees and reference to their position in the image
tiles_pos = []
tiles_pos_location = []
for tree in range(predictions_ide.shape[0]):
    if predictions_ide[tree] == 1:
        tiles_pos.append(tiles_large[tree, :])
        # this list contains reference to original tile :
        tiles_pos_location.append(tree)
tiles_pos = np.array(tiles_pos)
tiles_pos_location = np.array(tiles_pos_location)

# predict pixel position for each tree
model_pos = tf.keras.models.load_model(path_model_pos)
pred_pos = model_pos.predict(tiles_pos)

# get abs pixel locations of identified trees
pixels_pred, tile_info = \
    make_pixels_predicted(
        pos_list=tiles_pos_location, pred_pos_tiles=pred_pos, tile_info=tile_info, border=border)

draw_img_with_pixels(image, pixels_pred, report_path)


# ----------------------END--------------------------#
print("\n---------------------------------------------"
      "\nPrediction of trees is finished!")
