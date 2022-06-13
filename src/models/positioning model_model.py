"""
These functions represent modeling and training for tree Positioning of the pixel in a tile.
"""

#----------------------IMPORTS---------------------------#

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers

#------------------------FUNCTIONS-----------------------#


def make_model_ide(tile_size, border):



    return model, metrics


def train_ide(model_name, model, dataset_train, dataset_validate, epochs, path_model):

    # train model
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate,
    )

    # save model
    print(f"\nModel is trained and saved as {model_name}."
          f"   You can find the model in {path_model}."
          f"---To use it for future predictions - replace default!---")
    model.save(f"{path_model}{model_name}")

    return model, history


def render_history(model):
    plt.plot(model["loss"], label="loss")
    plt.plot(model["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Our losses of BinaryCrossentropy")
    plt.show()
    plt.close()

    plt.plot(model["F1Score"], label="F1Score")
    plt.plot(model["val_F1Score"], label="val_F1Score")
    plt.legend()
    plt.title("Our F1Score")
    plt.show()
    plt.close()


    plt.plot(model["precision"], label="precision")
    plt.plot(model["val_precision"], label="val_precision")
    plt.legend()
    plt.title("Our precision")
    plt.show()
    plt.close()
    pass


def perf_measure(tile_info, predictions):

    y_hat=predictions
    y_actual = tile_info[:, 2]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    num_tiles = len(y_actual)
    print(f"   TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"   TP: {int(TP/num_tiles*100)}%,  FP: {int(FP/num_tiles*100)}%,  TN: {int(TN/num_tiles*100)}%,  FN: {int(FN/num_tiles*100)}%")
    return(TP, FP, TN, FN)

