"""
This script will add an augmented version of an image.
"""

#----------------------INPUTS----------------------------#

image_path = '../data/raw/tree_detection/Potsdam/'
image_name = 'Potsdam_01'
suffix = '_aug_01'


#----------------------IMPORTS---------------------------#
from cachecontrol import CacheControl

import matplotlib.pyplot as plt
import os
from keras.utils import load_img
import imgaug as ia
import imgaug.augmenters as iaa

#---------------------FUNCTIONS--------------------------#
print(os.path.dirname(os.path.realpath(__file__)))

orig_image = plt.imread(image_path+image_name+'.jpg')

ia.seed(1)
seq = iaa.Sequential([
    iaa.LinearContrast((0.9, 1.1)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.15*1), per_channel=0.1),
], random_order=True) # apply augmenters in random order

images_aug = seq(images=image)
images_aug.save(image_path+image_name+suffix+'.jpg')

plt.imshow(image)
plt.title('Original image')
plt.show()
plt.close()
plt.imshow(images_aug)
plt.title('Augmented image')
plt.show()
