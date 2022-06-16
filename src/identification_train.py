"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  -------#

# Input specification:
## path with all raw data
path_input = '../data/raw/tree_detection/'
## targets: Name of specific image, position in json to be considered
targets = (["Potsdam_01", 0],
           ["Dresden_04", 0],
           ["Dresden_05", 0],
           ["Dresden_06", 0])
## number of augmented versions used for each image/target:
num_aug = 3

# Processing specification:
## in pixel:
tile_size = 25
## in pixel per side, expands small tile to large tile:
border = 15
## max number of neg per pos tiles in training set:
max_neg_per_true = 3

# Model specification:
path_model = '../models/ide_options/'
model_name = "ide_04"
path_model = path_model + model_name + '/'
batch_size = 30
epochs = 25

# -----------------------SETUP-----------------------#

print("\n--------------------------------------------"
      "\nStart training of IDENTIFICATION MODEL!")

import tensorflow as tf

from src.data.ide_loader \
    import images_labels_loader, input_creator
from src.models.identification_model \
    import make_model_ide, train_ide, render_history, perf_measure, ide_adj_pred


# ----------------DATA PREPARATION-----------#

# define data paths
path_raw_data, path_raw_json, name_raw_image, raw_image_number =\
    input_creator(
        targets, num_aug, path_input)
print(f"targets: {targets}"
      f"\npath_raw_data: {path_raw_data}"
      f"\npath_raw_json: {path_raw_json}"
      f"\nname_raw_image: {name_raw_image}")


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
