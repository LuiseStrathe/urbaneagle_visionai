"""
These functions load and transform data for tree identification.
"""

#----------------------IMPORTS---------------------------#

#import Augmentor
import matplotlib.pyplot as plt

from src.data.tree_data import *
from src.visualization.tree_vizualizations import *


#------------------------FUNCTIONS-----------------------#

def balance_tiles(tiles_large, tile_labels, max_neg_per_true):

    # concatenate tiles and labels
    #data_input = np.concatenate((tiles_large, tile_labels), axis=1)

    # define number of drops
    num_neg = np.array(np.where(tile_labels[:, 0] == 0)).shape[1]
    num_pos = np.array(np.where(tile_labels[:, 0] == 1)).shape[1]
    num_neg_target = int(num_pos * max_neg_per_true)
    num_neg_drops = num_neg - num_neg_target
    drop_rate = num_neg_drops / num_neg

    # create reduced input data
    tiles_input = []
    labels_input = []

    for tile in range(num_pos + num_neg):
        # random drops: 1 is keep, 0 is drop
        keep_or_drop = np.random.binomial(1, drop_rate)
        if (tile_labels[tile, 0] == 1) or (keep_or_drop == 1):
            tiles_input.append(tiles_large[tile])
            labels_input.append(tile_labels[tile])

    tiles_input = np.array(tiles_input)
    labels_input = np.array(labels_input)

    return tiles_input, labels_input

## --------------not in use -----------##
def augment_tiles(tiles_input, labels_input):

    # define augmentation size
    num_aug = int(2 * labels_input.shape[0])
    print(f"\n   Number of augmented tiles: {num_aug}")

    print(tiles_input.shape)
    print(labels_input.shape)

    # create augmented data
    p = Augmentor.DataPipeline(tiles_input, labels_input.all())
    #p.rotate(0.1, max_left_rotation=5, max_right_rotation=5)
    #p.flip_top_bottom(0.1)
    #p.flip_left_right(0.7)
    #p.shear(0.1, max_shear_left=5, max_shear_right=5)
    #p.skew_tilt(0.1, magnitude=0.1)
    tiles_new, labels_new = p.sample(num_aug)

    print(tiles_new.shape)
    print(labels_new.shape)

    plt.imshow(tiles_new[400])
    plt.title("augmented tile")
    plt.show()
    plt.imshow(tiles_new[1000])
    plt.title("augmented tile")
    plt.show()

    # concat and shuffle all data
    tiles_input = np.concatenate((tiles_input, tiles_new), axis=0)
    labels_input = np.concatenate((labels_input, tiles_new), axis=0)
    idx = np.random.permutation(labels_input.shape[0])
    tiles_input, labels_input = tiles_input[idx], labels_input[idx]
    print(f"   Number of tiles for training: {labels_input.shape[0]}\n")

    return tiles_new, labels_input


def make_directories(path_model):
    if not os.path.exists(f"{path_model}"):
        os.mkdir(f"{path_model}")
    if not os.path.exists(f"{path_model}/figures"):
        os.mkdir(f"{path_model}/figures")

    if not os.path.exists(f"{path_model}/images"):
        os.mkdir(f"{path_model}/images")



def images_loader(path_raw_data, name_raw_data, raw_image_number,
                  tile_size, border, batch_size, path_model, max_neg_per_true):

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

    # REBALANCE positive and negative tiles
    tiles_input, labels_input = balance_tiles(tiles_large, tile_labels, max_neg_per_true)

    # COMBINE tiles of images and add AUGMENTED
    #tiles_input, labels_input = augment_tiles(tiles_input, labels_input)

    # Make dataset
    dataset_train, dataset_validate = \
        make_train_set(tiles_input, labels_input, tile_size, border, batch_size)

    return  dataset_train, dataset_validate, \
            tiles_large, tile_info, tile_labels, tiles, tile_dims, \
            image, labels, orig_image