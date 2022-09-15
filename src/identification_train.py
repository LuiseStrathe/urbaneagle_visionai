"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  -------#

# Input specification:
## path with all raw data
path_input = '../data/input/tree_detection/'
## targets: Name of specific image, position in json to be considered
targets = (["Potsdam_01", 3, "labels_01.json"],
           ["Dresden_04", 1, "labels_01.json"],
           ["Dresden_05", 0, "labels_01.json"],
           ["Dresden_06", 2, "labels_01.json"])
## number of augmented versions used for each image/target:
num_aug = 1

# Processing specification:
## in pixel:
tile_size = 25
## in pixel per side, expands small tile to large tile:
border = 15
## max number of neg per pos tiles in training set:
max_neg_per_true = 3

# Model specification:
path_model = '../models/ide_options/'
model_name = "test"
path_model = path_model + model_name + '/'
batch_size = 25
epochs = 30

# -----------------------SETUP-----------------------#

print("\n--------------------------------------------"
      "\nStart training of IDENTIFICATION MODEL!")

import tensorflow as tf
import numpy as np
import gc

from src.data.ide_loader \
    import images_labels_loader, input_creator
from src.models.identification_model \
    import make_model_ide, train_ide, render_history, perf_measure, ide_adj_pred


# ----------------DATA PREPARATION-----------#

# define data paths
path_raw_data, path_raw_json, name_raw_image, raw_image_number =\
    input_creator(
        targets, num_aug, path_input)
print(f"\nInput Images: {targets}\n"
      f"path json: {path_raw_json}\n")

# loading images and labels and concatenate to training set
dataset_train, dataset_validate, tile_info = \
    images_labels_loader(
        path_raw_data, name_raw_image, raw_image_number, path_raw_json,\
        tile_size, border, batch_size, path_model, max_neg_per_true)

    
# ----------------TRAINING-------------------#

print("\n--------------------------------------------"
      "\nStart training of CNN...")

# training
model = \
    make_model_ide(
        tile_size, border)
model, history = \
    train_ide(
        model_name, model, dataset_train, dataset_validate, epochs, path_model)
del dataset_train, dataset_validate
gc.collect()

# ----------------SELF-PREDICTION-------------#

print("\n--------------------------------------------"
      "\nRunning evaluation")

# predict om given tiles
tiles_large = np.load(path_model + 'tiles_large.npy')
probabilities = \
    model.predict(
        tiles_large)
del model, tiles_large
gc.collect()

# optimize threshold and save predictions
tile_info[:, 3] = probabilities[:, 0]
threshold, predictions = \
    ide_adj_pred(
        probabilities, tile_info, path_model)
del probabilities
gc.collect()

render_history(history, path_model)
perf_measure(tile_info, predictions)


# ---------------END--------------------------#
tf.keras.backend.clear_session()
print("\n---------------------------------------------------------------"
      "\nTraining of IDENTIFICATION MODEL ended!")
