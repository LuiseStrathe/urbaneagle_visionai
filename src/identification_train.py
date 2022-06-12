"""
ABOUT
This script is used to train anew version of the IDENTIFICATION model.

INFO
Before start, check the initialization to modify the parameters.
"""



# ------------------------------------------------------------#
        ##   initialization    ##       change me  ##
# ------------------------------------------------------------#

# info about inputs: images and labels
path_raw_data = '../data/raw/tree_detection/Potsdam/'
name_raw_data = 'Potsdam_01'
## number the relevant image in the labeled data:
raw_image_number = 0
## in pixel:
tile_size = 25
## in pixel per side, expands small tile to large tile:
border = 15

# model specification
model_name = "ide_01"
batch_size = 300


# ------------------------------------------------------------#
        ##   setup    ##
# ------------------------------------------------------------#

# import libraries and functions
from src.data.data_loader import import_image
from src.visualization.tree_vizualizations import show_img_with_labels


# ------------------------------------------------------------#
        ##   data loading    ##
# ------------------------------------------------------------#
print("READY")

# import image
#image, labels, orig_image, labels_unscaled, i_width, i_height = \
 #   import_image(path_raw_data, name_raw_data, raw_image_number)
# import labels


# report on imports

#show_img_with_labels(image=orig_image, labels=labels, marker=',')


history_list = {}
