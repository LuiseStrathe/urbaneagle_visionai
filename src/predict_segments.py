"""
ABOUT
This script is used to segment a model into water, green, building and else areas.

INFO
Before start, check the initialization to modify the parameters.
"""

# ------------INITIALIZATION----  CHANGE ME  ----#

# info about inputs: images and labels
name = "Dresden_05_2022"
#image_path = f'../data/raw/images_unlabeled/{name}.jpg'
image_path = f'../data/raw/images_all/{name}.jpg'
report_path = '../reports/'
report_path = report_path + name + '/'

# model specification
tile_size = 512
path_model = '../models/'
model_name = "seg_512_opt"
path_model = path_model + model_name

# --------------------SETUP--------------------------#

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.data.seg_loader \
    import *

# ----------------DATA LOADING-----------------------#

print("\n--------------------------------------------"
      f"\nLoading Image for {name}...")

# prepare directories
if not os.path.exists(f"{report_path}"):
    os.mkdir(f"{report_path}")

# load data
image, tiles, dims = \
    pred_load(
        image_path, tile_size, name)

# -------------------Prediction-----------------#
print("\n--------------------------------------------"
      "\nStart prediction for segmentation...")

# predict the probabilities of each tile
model = tf.keras.models.load_model(path_model)
probabilities = ([])
for tile in tqdm(range(len(tiles))):
    prob = model.predict(np.array([tiles[tile, :, :, :]]))
    if tile == 0:
        probabilities = prob
    else: probabilities = np.concatenate((probabilities, prob), axis=0)
np.save(report_path + 'seg_prob.npy', probabilities)
print(f"probs sum:{probabilities[:, :, :,:].sum()}")
print(f"count of pixels (should be equal):{dims[1] * tile_size * dims[2] * tile_size}")


# concat the probability tiles
num_vertical = dims[1] * tile_size
num_horizontal = dims[2] * tile_size
prob_reshaped = ([])
for row in range(dims[1]):
  prob_row = ([])
  prob_row = np.concatenate([probabilities[(x + (row * dims[2])), :, :, :]
                                  for x in range(dims[2])], axis=1)
  if row == 0:
    prob_reshaped = prob_row
  else:
    prob_reshaped = np.concatenate((prob_reshaped, prob_row))

# make the prediction of the image
print("\nProcess to image...")
prediction = np.zeros(prob_reshaped.shape)
argmax_pred = np.argmax(prob_reshaped, axis=2)
for row in tqdm(range(num_vertical)):
  for col in range(num_horizontal):
      prediction[row, col, argmax_pred[row, col]] = 1.
np.save(report_path + "prediction.npy", prediction)


print(f"pred sum:{prediction.sum()}")
print(f"count of pixels (should be equal):{dims[1] * tile_size * dims[2] * tile_size}")

shares = np.array([prediction[:, :, x].sum() / prediction.sum() for x in range(4)])
print(f"area type shares:{shares}")
np.save(report_path + "area_shares.npy", shares)

prediction = prediction[:, :, :3]
print(f"New shape of prediction: {prediction.shape}")

im=prediction * 255
plt.imshow(im)
plt.savefig(report_path + "prediction.png")
plt.show()


# ----------------------END--------------------------#
print("\n---------------------------------------------"
      "\nPrediction of segments is finished!")
