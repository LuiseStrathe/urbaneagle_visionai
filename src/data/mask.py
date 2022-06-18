"""
This script will add a mask file for an image.
"""

#----------------------INPUTS----------------------------#

image_path = '../../data/raw/segmentation/'
image_name = 'Dresden_01'
input_prefix = '/task-10-annotation-9-by-1-tag-'
file_path_pre = image_path + image_name + input_prefix

# files
water = ["water-0.npy"]
green = ["green-0.npy", "green-1.npy", "green-2.npy", "green-3.npy"]
building = ["buildings-0.npy", "buildings-1.npy"]

#----------------------IMPORTS---------------------------#

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#---------------------EXECUTION--------------------------#

i_height, i_width = np.load(file_path_pre + water[0]).shape
mask = np.zeros((i_height, i_width, 4), dtype="float32")

for w in range(len(water)):
    mask[:, :, 0] += np.load(file_path_pre + water[w])
for g in range(len(green)):
    mask[:, :, 1] += np.load(file_path_pre + green[g])
for b in range(len(building)):
    mask[:, :, 2] += np.load(file_path_pre + building[b])

for i in tqdm(range(i_height)):
    for j in range(i_width):
        if (mask[i, j, 0] == 0) and (mask[i, j, 1] == 0) and (mask[i, j, 2] == 0):
            mask[i, j, 3] = 255

mask_image = mask[:, :, :3].copy()
mask /= 255.0

print(f"mask.shape = {mask.shape}")
np.save(image_path + image_name + "/mask.npy", mask)

plt.imshow(mask_image)
plt.show()
plt.close()

print("----- done -----")


