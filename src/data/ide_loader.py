"""
These functions load and transform data for tree identification.
"""

#----------------------IMPORTS---------------------------#

from src.data.tree_data import *
from src.visualization.tree_vizualizations import *

#------------------------FUNCTIONS-----------------------#

def make_directories(path_model):
    if not os.path.exists(f"{path_model}"):
        os.mkdir(f"{path_model}")
    if not os.path.exists(f"{path_model}/figures"):
        os.mkdir(f"{path_model}/figures")

    if not os.path.exists(f"{path_model}/images"):
        os.mkdir(f"{path_model}/images")

def images_loader(path_raw_data, name_raw_data, raw_image_number,
                  tile_size, border, batch_size, path_model):

    # prepare model directories
    make_directories(path_model)

    # LOAD images and labels
    image, labels, orig_image, labels_unscaled, i_width, i_height = \
        import_trees(path_raw_data, name_raw_data, raw_image_number, path_model)

    # SLICE TO TILES (=small tiles = moin tile reference)
    tiles, tile_info_initial, tile_dims = \
        make_tiles_small(image, name_raw_data, i_width, i_height, tile_size, border)

    # LABEL each tile
    tile_info, tile_labels = \
        label_tiles(labels, tile_info_initial, tile_size, tile_dims, border, name_raw_data, path_model)
    show_true_tiles(tile_info, name_raw_data, path_model)

    # EXPAND tiles to large tiles for the model as input
    tiles_large = \
        expand_tiles(tile_info, tile_dims, border, image, tile_size)

    # Make dataset
    dataset_train, dataset_validate = \
        make_train_set(tiles, tile_labels, tile_size, border, batch_size)

    return  dataset_train, dataset_validate, \
            tiles_large, tile_info, tile_labels, tiles, tile_dims, \
            image, labels, orig_image