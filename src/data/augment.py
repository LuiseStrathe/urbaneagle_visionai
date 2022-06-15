"""
This script will add an augmented version of an image.
"""

#----------------------INPUTS----------------------------#

image_path = '../../data/raw/tree_detection/Potsdam/'
image_name = 'Potsdam_01'
suffix = '_aug_03'


#----------------------IMPORTS---------------------------#

import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image


#---------------------FUNCTIONS--------------------------#

# prepare image and augmentation
image = plt.imread(f'{image_path}{image_name}.jpg')

transform = A.Compose(
    [A.HueSaturationValue(
      always_apply=True),
     A.Sharpen(
      alpha=(0, 0.2), lightness=(0.8, 1.0), always_apply=True),
     A.RandomBrightnessContrast(
      brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True),
     A.RGBShift(
      r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=False)
     ])

# transform image and safe under given file name
augmented_image = transform(image=image)['image']
img = Image.fromarray(augmented_image, 'RGB')
img.save(f'{image_path}{image_name}{suffix}.jpg')
print(augmented_image.shape)


