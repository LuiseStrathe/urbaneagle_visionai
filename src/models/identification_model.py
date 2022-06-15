"""
These functions represent modeling and training for tree identification.
"""

#----------------------IMPORTS---------------------------#

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers, optimizers

#------------------------FUNCTIONS-----------------------#


def make_model_ide(tile_size, border, batch_size):

    tile_size = tile_size + 2 * border

    metrics = [
      tf.keras.metrics.BinaryAccuracy(name='binary_crossentropy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Precision(name='binary_accuracy'),
      tf.keras.metrics.Recall(name='F1Score'),
      ]
    optimizer = optimizers.Adam()
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    model = models.Sequential()
    model.add(layers.Input(shape=(tile_size, tile_size, 3)))
    model.add(layers.Conv2D(20, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(30, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(150, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer=initializer))
    model.summary()

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=metrics,
        )

    return model, metrics


def train_ide(model_name, model, dataset_train, dataset_validate, epochs, path_model):

    # Model weights are saved at the end of every epoch,
        # if it's the best seen so far.
    checkpoint_filepath = '../models/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='F1Score',
        mode='max',
        save_best_only=True)

    # train model
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate,
        callbacks=[model_checkpoint_callback],
    )

    # The model weights (that are considered the best)
    #  are loaded into the model.
    model.load_weights(checkpoint_filepath)


    # save model
    print(f"\nModel is trained and saved as {model_name}."
          f"   You can find the model in {path_model}."
          f"---To use it for future predictions - replace default!---")
    model.save(f"{path_model}")

    return model, history


def render_history(history, path_model):

    metrics = (["binary_crossentropy", "F1Score", "precision", "binary_accuracy"])

    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.legend()
        plt.title(f"Our losses of {metric}")
        plt.show()
        plt.savefig(f"{path_model}figures/{metric}.png")

    pass


def ide_adj_pred(probabilities, tile_info, path_model):

    # optimal threshold
    num_orig_pos = len(np.where(tile_info[:, 2] == 1)[0])
    print(  f"Number of tiles: {tile_info.shape[0]}"
            f"\nNumber of true pos tiles for evaluation: {num_orig_pos}")
    threshold = 0.1
    for thresh in np.arange(0., 1., 0.01):
        thresh = round(thresh, 2)
        num_pred_labels = len(np.where(probabilities > thresh)[0])
        print(f"Thresh: {thresh}"
              f"\nNumber of predicted labels: {num_pred_labels}")
        if num_pred_labels < num_orig_pos:
            threshold = thresh
            break

    # save predictions and tile_info
    predictions = np.array([1 if x >= threshold else 0 for x in probabilities])
    threshold = np.array([threshold])
    np.savez(f"{path_model}saved_training_info.npz",
             threshold=threshold, predictions=predictions, tile_info=tile_info)

    print("\nOptimal threshold:", threshold[0])

    return threshold, predictions


def perf_measure(tile_info, predictions):

    y_hat = predictions
    y_actual = tile_info[:, 2]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1

    num_tiles = len(y_actual)

    print(f"   TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n"
          f"   TP: {int(TP/num_tiles*100)}%,  "
          f"   FP: {int(FP/num_tiles*100)}%,  "
          f"   TN: {int(TN/num_tiles*100)}%,  "
          f"   FN: {int(FN/num_tiles*100)}%")

    return(TP, FP, TN, FN)

