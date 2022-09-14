"""
These functions load and transform data for segmentation.
"""

#----------------------IMPORTS---------------------------#

import gc
import os
import numpy as np
import tensorflow as tf
from keras.utils import load_img
from keras.utils import img_to_array


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


def make_tiles(
        x, image, mask, target, i_width, i_height, tile_size):

    # prepare variables for both tile layers
    ## rounded down to full tiles ignoring the last tile if it is not full:
    num_hor = int(i_width // tile_size)
    num_ver = int(i_height // tile_size)
    num_tiles = num_hor * num_ver
    print(  f"Slicing image {target}\n"
            f"   There are {num_tiles} small tiles at {tile_size}x{tile_size} pixels\n"
            f"   Small tiles identified: {num_ver} X {num_hor} tiles")

    # prepare variables for the tiles
    image = np.array(image) # input image
    tiles = np.zeros((num_tiles, tile_size, tile_size, 3)) # all tile images
    masks = np.zeros((num_tiles, tile_size, tile_size, 4)) # all tile masks

    # initiate info variables
    ## store information about the tiles in total and in each layer:
    tile_dims = (num_tiles, num_ver, num_hor)
    tile_info = np.zeros((num_tiles, 5)) # store information about the tiles
    ## tile_info:
        ### [0:2] image_number, tile row, tile column,
        ### [3:4] position horizontal, position vertical  // (top left corner of tile in the image)

    # create tiles and masks for target image
    for i in range(num_ver):
        for j in range(num_hor):
            tile_num = i * num_hor + j
            # give starting (left/top) & mean pixels in rows and columns for this tile
            v_start = i * tile_size
            h_start = j * tile_size
            # create tile
            tile = image[v_start:v_start+tile_size, h_start:h_start+tile_size, :]
            mask_tile = mask[v_start:v_start+tile_size, h_start:h_start+tile_size, :]
            # add to array
            tiles[tile_num] = tile
            masks[tile_num] = mask_tile
            # store location ot tiles:
            tile_info[tile_num] = (x, i, j, v_start, h_start)

    print(f"   Tiles created: {tiles.shape}.\n")
    print(f"   Masks created: {masks.shape}.\n")

    return tiles, masks, tile_info, tile_dims


def image_mask_load(
        path_input, target, x, tile_size):

    # load image and mask
    image = load_img(path_input + target + "/" + target + '.jpg')
    image = img_to_array(image)
    image = image.astype('float32')
    i_height, i_width, channels = image.shape
    image /= 255.0

    mask = np.load(path_input + target + "/mask.npy")

    print(f"\nInput Images: {target}\n"
          f"image.shape: {image.shape}\n"
          f"masks.shape: {mask.shape}\n")

    # SLICE to tiles
    x_tiles, x_masks, x_tile_info, x_dims = \
        make_tiles(
            x, image, mask, target, i_width, i_height, tile_size)

    del mask, image
    gc.collect()

    return x_tiles, x_masks, x_tile_info, x_dims


def make_train_set(
        tiles, masks, tile_size, batch_size):

    # define the training set
    train_ratio = 0.8
    num_tiles = tiles.shape[0]
    num_train_tiles = int(num_tiles * train_ratio)
    num_val_tiles = num_tiles - num_train_tiles
    print(f"There are {num_tiles} tiles in total.")
    print(f"There are {num_train_tiles} tiles in the training set.")
    print(f"There are {num_val_tiles} tiles in the validation set.")

    # split into training and validation set
    train_tiles = tiles[:int(num_tiles*train_ratio)]
    train_masks = masks[:int(num_tiles*train_ratio)]
    val_tiles = tiles[int(num_tiles*train_ratio):]
    val_masks = masks[int(num_tiles*train_ratio):]
    print(f"Training set: {train_tiles.shape} - ({train_ratio} share of all tiles)")
    print(f"Validation set: {val_tiles.shape}")

    # build tensorflow dataset
    dataset_train_original = tf.data.Dataset.from_tensor_slices((train_tiles, train_masks))
    dataset_validate_original = tf.data.Dataset.from_tensor_slices((val_tiles, val_masks))

    # shuffle and batch
    def encode(tile, mask):
        image_encoded = tf.image.convert_image_dtype(tile, dtype=tf.float32)
        image_encoded = tf.image.resize(image_encoded, (tile_size, tile_size))
        return image_encoded, mask

    dataset_train = dataset_train_original.map(
        lambda image, label: encode(image, label)).cache().shuffle(2500).batch(batch_size)
    dataset_validate = dataset_validate_original.map(
        lambda image, label: encode(image, label)).cache().batch(batch_size)

    return dataset_train, dataset_validate


def seg_train_load(
        path_input, targets,
        tile_size, batch_size, path_model):

    # make the model_name directory
    if not os.path.exists(f"{path_model}"):
        os.mkdir(f"{path_model}")


    #----------------IMAGE-LOOP----------------------#

    for x in range(len(targets)):

        # LOAD image and labels
        x_tiles, x_masks, x_tile_info, x_dims = \
            image_mask_load(
                path_input, targets[x], x, tile_size)

        # concat data of image x
        if x == 0:
            tiles = x_tiles
            masks = x_masks
            tile_info = x_tile_info
            dims = x_dims
        else:
            tiles = np.concatenate((tiles, x_tiles), axis=0)
            masks = np.concatenate((masks, x_masks), axis=0)
            tile_info = np.concatenate((tile_info, x_tile_info), axis=0)
            dims = np.concatenate((dims, x_dims), axis=0)

    del x_tiles, x_masks, x_tile_info, x_dims
    gc.collect()

    # ----------------TRAINING SET------------------#

    # Make dataset
    dataset_train, dataset_validate = \
        make_train_set(
            tiles, masks, tile_size, batch_size)

    del masks
    gc.collect()

    return tiles, dataset_train, dataset_validate, tile_info, dims


def pred_load(
        image_path, tile_size, name):

    image = load_img(image_path)
    image = img_to_array(image)
    image = image.astype('float32')
    i_height, i_width, channels = image.shape
    image /= 255.0

    # SLICE to tiles
    # rounded down to full tiles ignoring the last tile if it is not full:
    num_hor = int(i_width // tile_size)
    num_ver = int(i_height // tile_size)
    num_tiles = num_hor * num_ver
    dims = [num_tiles, num_ver, num_hor]
    print(f"Slicing image {name}\n"
          f"   There are {num_tiles} small tiles at {tile_size}x{tile_size} pixels\n"
          f"   Small tiles identified: {num_ver} X {num_hor} tiles")

    # prepare variables for the tiles
    image = np.array(image)  # input image
    tiles = np.zeros((num_tiles, tile_size, tile_size, 3))  # all tile images

    # create tiles and masks for target image
    for i in range(num_ver):
        for j in range(num_hor):
            tile_num = i * num_hor + j
            # give starting (left/top) & mean pixels in rows and columns for this tile
            v_start = i * tile_size
            h_start = j * tile_size
            # create tile
            tile = image[v_start:v_start + tile_size, h_start:h_start + tile_size, :]
            # add to array
            tiles[tile_num] = tile

    print(f"   Tiles created: {tiles.shape}.\n")

    return image, tiles, dims