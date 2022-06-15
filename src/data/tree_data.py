"""
These functions load and transform data for tree identification.
"""

#----------------------IMPORTS---------------------------#

import numpy as np
import json

import tensorflow as tf
from keras.utils import load_img
from keras.utils import img_to_array
from matplotlib import pyplot as plt


#------------------------FUNCTIONS-----------------------#

def import_trees(path_raw_data, name_raw_data, raw_image_number, path_model, path_raw_json):

    print("\n---------"
          f"\nImporting image: {name_raw_data} (exact item in upload: {raw_image_number})")

    # -------------------EXECUTION---------------------#

    # load the image to get its shape
    orig_image = load_img(path_raw_data+name_raw_data+'.jpg')
    i_width, i_height = orig_image.size

    # convert to numpy array
    image = img_to_array(orig_image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # load the label-points for the picture from label studio
    with open(path_raw_json+'.json') as json_file:
        json_load = json.load(json_file)

    # select data
    json_load = json_load[raw_image_number]["annotations"][0]["result"]
    labels_x = ([])
    labels_y = ([])
    for i in json_load:
        labels_x = np.append(labels_x, i["value"]["x"])
        labels_y = np.append(labels_y, i["value"]["y"])
    labels_unscaled = np.vstack((labels_y, labels_x)).transpose()

    # scale data points to image pixels
    reshaper = [[(i_height/100), 0], [0, (i_width/100)]]
    labels = np.mat(labels_unscaled) * np.mat(reshaper)

    # add row for dedicated tile
    labels = labels.astype('int')
    labels = np.hstack((labels, 0*labels[:, 0]))

    # ---------------------INFO-------------------------#

    # show image with labels
    plt.imshow(orig_image)
    for tree in range(labels[:,0].shape[0]):
        plt.plot(labels[tree, 1], labels[tree, 0], '+', color="purple", markersize=5)
    plt.title(f"{name_raw_data} image and original tree labels\n"
              f"   There are {labels_unscaled[:,0].shape} labeled trees in the picture"
              f"\n   The image size is {i_height}x{i_width} pixels."
              f"\n   The {len(labels_x)} data points are distributed along the respective axis as follows:")
    plt.savefig(f'{path_model}figures/img_w_labels.png')
    plt.show()
    plt.close()

    # check label distribution in histogram
    plt.hist((labels[:, 1]), bins=200)
    plt.hist((labels[:, 0]), bins=200)
    plt.title("Label distribution along each axis:")
    plt.savefig(f'{path_model}figures/label_distribution.png')
    plt.show()

    return image, labels, orig_image, labels_unscaled, i_width, i_height


def make_tiles_small(image, image_name, i_width, i_height, tile_size, border):

    # prepare variables for both tile layers
    ## rounded down to full tiles ignoring the last tile if it is not full:
    num_hor = int((i_width - 2 * border) // tile_size)
    num_ver = int((i_height - 2 * border) // tile_size)
    num_tiles = num_hor * num_ver
    print(  f"Slicing image {image_name}\n"
            f"   There are {num_tiles} small tiles at {tile_size}x{tile_size} pixels\n"
            f"   Small tiles identified: {num_ver} X {num_hor} tiles")

    # prepare variables for the tiles
    image = np.array(image) # input image
    tiles = np.zeros((num_tiles, tile_size, tile_size, 3)) # all tile images

    # initiate info variables
    ## store information about the tiles in total and in each layer:
    tile_dims = (num_tiles, num_ver, num_hor)
    tile_info_initial = np.zeros((num_tiles, 8)) # store information about the tiles
    ## tile_info:
        ### [0:3] tile row, tile column, [true_label], probability,
        ### [4:5] position horizontal, position vertical  // (top left corner of tile in the image)
        ### [6:7] vertical label position, horizontal label position within the (small) tile

    # create tiles
    for i in range(num_ver):
        for j in range(num_hor):
            tile_num = i * num_hor + j
            # give starting (left/top) & mean pixels in rows and columns for this tile
            v_start = border + i * tile_size
            h_start = border + j * tile_size
            # create tile
            tile = image[v_start:v_start+tile_size, h_start:h_start+tile_size, :]
            # add tile to array
            tiles[tile_num] = tile
            # store location ot tiles:
            tile_info_initial[tile_num] = (i,j,0,0,v_start, h_start, 0, 0)

    print(f"   Tiles created: {tiles.shape}.\n")
    return tiles, tile_info_initial, tile_dims


def label_tiles(
        labels, tile_info_initial, tile_size,
        tile_dims, border, name_raw_data, path_model):

    # prepare variables
    num_tiles = tile_dims[0]
    max_v = int(tile_size * tile_dims[1] - 2 * border)
    max_h = int(tile_size * tile_dims[2] - 2 * border)
    tile_labels = np.zeros((num_tiles, 1))
        ## labels for each tile: 0 = no tree, 1 = tree
    tile_dense = np.zeros((num_tiles, 1))
        ## number of trees in each tile = density of a tile
    tile_info = tile_info_initial

    # Labeling the tiles
    for label in labels:
        # ignore if out of image range
        l_v = label[:, 0]
        l_h = label[:, 1]

        if (l_v<=max_v and l_v>border) and (l_h<=max_h and l_h>border):
            # find the tile that contains the label
            ## step through horizontal tiles alternating between layers
            pos_vertical = (l_v - border) // tile_size  # position in vertical direction
            pos_horizontal = (l_h - border) // tile_size  # position in horizontal direction
            tile_num = int((pos_vertical * tile_dims[2]) + pos_horizontal)

            # update arrays
            tile_labels[tile_num] = 1
            tile_dense[tile_num] += 1
            tile_info[tile_num][2] = 1
            tile_info[tile_num][6] = l_v - tile_info[tile_num][4]
            tile_info[tile_num][7] = l_h - tile_info[tile_num][5]
            label[:, 2] = tile_num


    # show label distribution amongst tiles
    plt.hist(tile_dense, bins=5, color="green")
    plt.title(f"{name_raw_data}: Distribution of trees per tile:\n"
              f"   The max number of trees per tile is: {tile_dense.max()}\n"
              f"   Labels vector is created over all tiles: {tile_labels.shape}.\n"
              f"   Total trees labeled in image: {tile_dense.sum()}.\n"
              f"   Tiles labeled as with tree: {tile_labels.sum()}\n"
              f"   This is {tile_labels.sum() / tile_labels.shape[0] * 100}% of the tiles.\n"
              )
    plt.savefig(f'{path_model}figures/info_import.png')
    plt.show()
    plt.close()

    return tile_info, tile_labels


def expand_tiles(tile_info, tile_dims, border, image, tile_size):

    image = np.array(image)
    tile_size_large = tile_size + 2 * border
    tiles_large = np.zeros((tile_dims[0], tile_size_large, tile_size_large, 3))

    for tile in range(tile_dims[0]):
        v_start = int(tile_info[tile, 4] - border)
        h_start = int(tile_info[tile, 5] - border)
        tile_large = image[v_start:v_start+tile_size_large, h_start:h_start+tile_size_large, :]
        tiles_large[tile] = tile_large
    return tiles_large


def make_train_set(tiles_large, tile_labels, tile_size, border, batch_size):

    # define the training set
    tile_size = (tile_size + 2 * border)
    train_ratio = 0.8
    num_tiles = tile_labels.shape[0]
    num_train_tiles = int(num_tiles * train_ratio)
    num_val_tiles = num_tiles - num_train_tiles
    print(f"There are {num_tiles} tiles in total.")
    print(f"There are {num_train_tiles} tiles in the training set.")
    print(f"There are {num_val_tiles} tiles in the validation set.")

    # split into training and validation set
    train_tiles = tiles_large[:int(num_tiles*train_ratio)]
    train_labels = tile_labels[:int(num_tiles*train_ratio)]
    val_tiles = tiles_large[int(num_tiles*train_ratio):]
    val_labels = tile_labels[int(num_tiles*train_ratio):]
    print(f"Training set: {train_tiles.shape} - ({train_ratio} share of all tiles)")
    print(f"Validation set: {val_tiles.shape}")

    # build tensorflow dataset
    dataset_train_original = tf.data.Dataset.from_tensor_slices((train_tiles, train_labels))
    dataset_validate_original = tf.data.Dataset.from_tensor_slices((val_tiles, val_labels))

    # shuffle and batch
    def encode(tile, label):
        image_encoded = tf.image.convert_image_dtype(tile, dtype=tf.float32)
        image_encoded = tf.image.resize(image_encoded, (tile_size, tile_size))
        return image_encoded, label

    dataset_train = dataset_train_original.map(
        lambda image, label: encode(image, label)).cache().shuffle(2500).batch(batch_size)
    dataset_validate = dataset_validate_original.map(
        lambda image, label: encode(image, label)).cache().batch(batch_size)

    return dataset_train, dataset_validate