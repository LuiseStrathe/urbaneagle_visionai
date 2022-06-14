"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""



# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
path_raw_data = '../data/raw/tree_detection/Potsdam/'
name_raw_data = 'Potsdam_01'
    ## number the relevant image in the labeled data:
raw_image_number = 0
    ## in pixel:
tile_size = 25
    ## in pixel per side, expands small tile to large tile:
border = 2
    ## max number of neg per pos tiles in training set:
max_neg_per_true = 2

# model specification
path_model = '../models/ide_options/'
model_name = "ide_03"
path_model = path_model + model_name + '/'
batch_size = 100
epochs = 2


# -----------------------SETUP-----------------------#

print("\n--------------------------------------------"
      "\nStart training of IDENTIFICATION MODEL!")

import tensorflow as tf

from src.data.ide_loader \
    import images_loader
from src.models.identification_model \
    import make_model_ide, train_ide, render_history, perf_measure, ide_adj_pred


# ----------------DATA PREPARATION-----------#

# loading images and labels and concatenate to training set
dataset_train, dataset_validate, \
            tiles_large, tile_info, tile_labels, tiles, tile_dims, \
            image, labels, orig_image = \
            images_loader(
                path_raw_data, name_raw_data, raw_image_number, \
                tile_size, border, batch_size, path_model, max_neg_per_true)


# ----------------TRAINING-------------------#

print("\nStart training of CNN...")

# training
model, metrics = make_model_ide(tile_size, border, batch_size)
model, history = \
    train_ide(model_name, model, dataset_train, dataset_validate, epochs, path_model)


# ----------------SELF-PREDICTION-------------#

# predict om given tiles
probabilities = model.predict(tiles_large)

# optimize threshold and save predictions
tile_info[:, 3] = probabilities[:, 0]
threshold, predictions = ide_adj_pred(probabilities, tile_info, path_model)


# ---------------PERFORMANCE------------------#

print("\nRunning evaluation")
render_history(history, path_model)
perf_measure(tile_info, predictions)

tf.keras.backend.clear_session()


print("\n---------------------------------------------------------------"
      "\nTraining of IDENTIFICATION MODEL ended!")



