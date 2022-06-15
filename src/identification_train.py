"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
path_raw_data = (['../data/raw/tree_detection/Potsdam/',
                  '../data/raw/tree_detection/Potsdam/',
                  '../data/raw/tree_detection/Potsdam/',
                  '../data/raw/tree_detection/Potsdam/',
                  '../data/raw/tree_detection/Potsdam/'])
path_raw_json = (['../data/raw/tree_detection/Potsdam/Potsdam_01',
                  '../data/raw/tree_detection/Potsdam/Potsdam_01',
                  '../data/raw/tree_detection/Potsdam/Potsdam_01',
                  '../data/raw/tree_detection/Potsdam/Potsdam_01',
                  '../data/raw/tree_detection/Potsdam/Potsdam_01'])
name_raw_image = (['Potsdam_01',
                   'Potsdam_01_aug_01',
                   'Potsdam_01_aug_02',
                   'Potsdam_01_aug_03',
                   'Potsdam_01_aug_04'])
## number the relevant image in the labeled data:
raw_image_number = ([0, 0, 0, 0, 0])

## in pixel:
tile_size = 25
## in pixel per side, expands small tile to large tile:
border = 15
## max number of neg per pos tiles in training set:
max_neg_per_true = 3

# model specification
path_model = '../models/ide_options/'
model_name = "ide_03"
path_model = path_model + model_name + '/'
batch_size = 40
epochs = 25

# -----------------------SETUP-----------------------#

print("\n--------------------------------------------"
      "\nStart training of IDENTIFICATION MODEL!")

import tensorflow as tf

from src.data.ide_loader \
    import images_labels_loader
from src.models.identification_model \
    import make_model_ide, train_ide, render_history, perf_measure, ide_adj_pred


# ----------------DATA PREPARATION-----------#

# loading images and labels and concatenate to training set
dataset_train, dataset_validate, tiles_large, tile_info = \
    images_labels_loader(
        path_raw_data, name_raw_image, raw_image_number, path_raw_json,\
        tile_size, border, batch_size, path_model, max_neg_per_true)


# ----------------TRAINING-------------------#

print("\n--------------------------------------------"
      "\nStart training of CNN...")

# training
model, metrics = \
    make_model_ide(
        tile_size, border, batch_size)
model, history = \
    train_ide(
        model_name, model, dataset_train, dataset_validate, epochs, path_model)


# ----------------SELF-PREDICTION-------------#

print("\n--------------------------------------------"
      "\nRunning evaluation")

# predict om given tiles
probabilities = \
    model.predict(
        tiles_large)

# optimize threshold and save predictions
tile_info[:, 3] = probabilities[:, 0]
threshold, predictions = \
    ide_adj_pred(
        probabilities, tile_info, path_model)

render_history(history, path_model)
perf_measure(tile_info, predictions)


# ---------------END--------------------------#
tf.keras.backend.clear_session()
print("\n---------------------------------------------------------------"
      "\nTraining of IDENTIFICATION MODEL ended!")
