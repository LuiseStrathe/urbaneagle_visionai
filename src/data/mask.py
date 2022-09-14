"""
This script will add a mask file for an image.
"""

#----------------------INPUTS----------------------------#

image_path = '../../data/raw/segmentation/'
image_name = 'Berlin_01'
input_prefix = '/task-12-annotation-10-by-1-tag-'
file_path_pre = image_path + image_name + input_prefix

# files as RGB
list_building = ["buildings-0.npy", "buildings-1.npy", "buildings-2.npy", "buildings-3.npy"]
list_green = ["green-0.npy"]
list_water = ["water-0.npy", "water-1.npy"]

#----------------------IMPORTS---------------------------#

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#---------------------EXECUTION--------------------------#

i_height, i_width = np.load(file_path_pre + list_water[0]).shape
mask = np.zeros((i_height, i_width, 4), dtype="float32")

building = np.array([np.load(file_path_pre + file) for file in list_building], dtype="float32")
green = np.array([np.load(file_path_pre + file) for file in list_green], dtype="float32")
water = np.array([np.load(file_path_pre + file) for file in list_water], dtype="float32")

print(f"buildings: {building.shape}, max:{building.max()}, sum:{building.sum()}")
print(f"green: {green.shape}, max:{green.max()}, sum:{green.sum()}")
print(f"water: {water.shape}, max:{water.max()}, sum:{water.sum()}")
print(f"pixels in total {i_height * i_width}\n")

for i in tqdm(range(i_height)):
    for j in range(i_width):
        if np.array([building[b][i][j] for b in range(len(list_building))]).sum() > 0:
                mask[i][j][0] = 1.
        elif np.array([green[g][i][j] for g in range(len(list_green))]).sum() > 0:
                mask[i][j][1] = 1.
        elif np.array([water[w][i][j] for w in range(len(list_water))]).sum() > 0:
                mask[i][j][2] = 1.
        else: mask[i][j][3] = 1.

plt.imshow(mask[:, :, :3])
plt.show()
plt.savefig(image_path + image_name + "_mask.png")
np.save(image_path + image_name + "/mask.npy", mask)

print(f"Mask imported: {mask.shape}.\n"
      f"class distribution: {[np.sum(mask[: , :, i]) / mask[:, :, 0].size for i in range(4)]}")

print("----- done -----")


