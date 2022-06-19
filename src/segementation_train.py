"""
ABOUT
This script is used to train a new version of the SEGMENTATION model.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  -------#

# Input specification:
# targets: Name of specific image:
targets = (["Dresden_01"])
path_input = '../data/raw/segmentation/'
# number of augmented versions used for each image/target:
num_aug = 2

# Define tiles (quadrants) to be used for training:
tile_size = 512

# Model specification:
path_model = '../models/segmentation_options/'
model_name = f"seg_{tile_size}_new"
path_model = path_model + model_name + '/'
batch_size = 2
epochs = 10


# -----------------------SETUP-----------------------#

print("\n--------------------------------------------"
      "\nStart training of SEGMENTATION MODEL!")

import tensorflow as tf
import numpy as np
import os
import gc
from tqdm import tqdm

from src.data.seg_loader \
    import *

# ----------------DATA PREPARATION-----------#
tiles, dataset_train, dataset_validate, tile_info, dims =\
    seg_train_load(
        path_input, targets,
        tile_size, batch_size, path_model)

    
# ----------------TRAINING-------------------#

print("\n--------------------------------------------"
      "\nStart training of U-CNN...")

# load pre-build resnet50 unet
model = tf.keras.models.load_model(path_model)
model.summary()

metrics = [
      tf.keras.metrics.BinaryAccuracy(name='binary_crossentropy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Precision(name='binary_accuracy'),
      tf.keras.metrics.Recall(name='F1Score'),
      ]
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer="Adam",
    metrics=metrics,
    )

# Model weights are saved at the end of every epoch
checkpoint_filepath = '../models/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='F1Score',
    mode='max',
    save_best_only=True)

# training
# train model
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_validate,
    callbacks=[model_checkpoint_callback],
)
del dataset_train, dataset_validate
gc.collect()

# save model
model.load_weights(checkpoint_filepath)
print(f"\nModel is trained and saved as {model_name}."
      f"   You can find the model in {path_model}."
      f"---To use it for future predictions - replace default!---")
model.save(f"{path_model}")

# ----------------SELF-PREDICTION-------------#

print("\n--------------------------------------------"
      "\nRunning evaluation")

# predict om given tiles
probabilities = \
    model.predict(
        tiles)
print(probabilities.shape)

del model, tiles
gc.collect()

# optimize threshold and save predictions
#tile_info[:, 3] = probabilities[:, 0]
#predictions = \
#    seg_adj_pred(
#        probabilities, tile_info, path_model)
#del probabilities
gc.collect()

def render_history(history, path_model):
    import matplotlib.pyplot as plt
    metrics = (["binary_crossentropy", "F1Score", "precision", "binary_accuracy"])

    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.legend()
        plt.title(f"Our losses of {metric}")
        plt.show()
        plt.savefig(f"{path_model}{metric}.png")

    pass
render_history(history, path_model)


# ---------------END--------------------------#
tf.keras.backend.clear_session()
print("\n---------------------------------------------------------------"
      "\nTraining of SEGMENTATION MODEL ended!")
