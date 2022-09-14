"""
These functions load and transform data for tree identification.
"""

#----------------------IMPORTS---------------------------#

import gc

from src.data.tree_data import *
from src.visualization.tree_vizualizations import *
from src.helper.helper import make_directories


#------------------------FUNCTIONS-----------------------#

def input_creator(targets, num_aug, path_input):

    path_raw_data = ([])
    path_raw_json = ([])
    name_raw_image = ([])
    raw_image_number = ([])

    for target, number, json in targets:
        image_path = path_input + target[:-3] + '/'

        [path_raw_data.append(image_path)
         for i in range(num_aug + 1)]
        [path_raw_json.append(path_input + json)
         for i in range(num_aug + 1)]
        name_raw_image.append(target)
        [name_raw_image.append(target + '_aug_0' + str(i + 1))
         for i in range(num_aug)]
        [raw_image_number.append(number)
         for i in range(num_aug + 1)]


    return path_raw_data, path_raw_json, name_raw_image, raw_image_number


def balance_tiles(
        tiles_large, tile_labels, max_neg_per_true):

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
        if (tile_labels[tile, 0] == 1) or (keep_or_drop == 0):
            tiles_input.append(tiles_large[tile])
            labels_input.append(tile_labels[tile])

    tiles_input = np.array(tiles_input)
    labels_input = np.array(labels_input)

    print(f"\nInitial number of tiles: {num_pos + num_neg}\n"
          f"Target for neg drops of tiles: ~{num_neg_drops}\n"
          f"Final number of tiles: {tiles_input.shape[0]}\n"
          f"Final positive: {np.array(np.where(labels_input[:, 0] == 1)).shape[1]}\n"
          f"Final negative: {np.array(np.where(labels_input[:, 0] == 0)).shape[1]}\n")

    return tiles_input, labels_input


def image_label_load(
        path_raw_data, name_raw_image, raw_image_number, path_model, path_raw_json,
        tile_size, border):

    # LOAD image and labels
    image, labels, labels_unscaled, i_width, i_height = \
        import_trees(
            path_raw_data, name_raw_image, raw_image_number, path_model, path_raw_json)

    # SLICE to tiles (=small tiles = moin tile reference)
    tiles, tile_info_initial, tile_dims = \
        make_tiles_small(
            image, name_raw_image, i_width, i_height, tile_size, border)

    # LABEL each tile
    x_tile_info, x_labels_input = \
        label_tiles(
            labels, tile_info_initial, tile_size, tile_dims, border, name_raw_image, path_model)
    show_true_tiles(
        x_tile_info, name_raw_image, path_model)

    # discard unnecessary data
    del labels, labels_unscaled, tile_info_initial
    gc.collect()

    # EXPAND tiles to large tiles for the model as input
    x_tiles_large = \
        expand_tiles(x_tile_info, tile_dims, border, image, tile_size)

    # discard unnecessary data
    del image
    gc.collect()

    return x_tiles_large, x_labels_input, x_tile_info


def images_labels_loader(
        path_raw_data, name_raw_image, raw_image_number, path_raw_json,
        tile_size, border, batch_size, path_model, max_neg_per_true):

    # prepare directories
    make_directories(path_model)


    #----------------IMAGE-LOOP----------------------#

    for x in range(len(raw_image_number)):

        # LOAD image and labels
        x_tiles_large, x_labels_input, x_tile_info = \
            image_label_load(
                path_raw_data[x], name_raw_image[x], raw_image_number[x],  path_model, path_raw_json[x],
                tile_size, border)

        # concat data of image x
        if x == 0:
            tiles_large = x_tiles_large
            labels_input = x_labels_input
            tile_info = x_tile_info
        else:
            tiles_large = np.concatenate((tiles_large, x_tiles_large), axis=0)
            labels_input = np.concatenate((labels_input, x_labels_input), axis=0)
            tile_info = np.concatenate((tile_info, x_tile_info), axis=0)

    del x_tiles_large, x_labels_input, x_tile_info
    tiles_input = tiles_large
    np.save(path_model + 'tiles_large', tiles_large)
    del tiles_large
    gc.collect()


    # ----------------TRAINING SET------------------#

    # REBALANCE positive and negative tiles
    tiles_input, labels_input = \
        balance_tiles(
            tiles_input, labels_input, max_neg_per_true)

    # shuffle all data
    idx = np.random.permutation(labels_input.shape[0])
    tiles_input, labels_input = tiles_input[idx], labels_input[idx]


    # Make dataset
    dataset_train, dataset_validate = \
        make_train_set(
            tiles_input, labels_input, tile_size, border, batch_size)

    # discard unnecessary data
    del labels_input, tiles_input, idx
    gc.collect()


    return  dataset_train, dataset_validate, tile_info