"""
These functions load and transform data for tree prediction.
"""

#----------------------IMPORTS---------------------------#
import os
from src.data.tree_data import *


#------------------------FUNCTIONS-----------------------#


def image_load_tiles(
        image_path, tile_size, border, name, report_path):

    # prepare directories
    if not os.path.exists(f"{report_path}"):
        os.mkdir(f"{report_path}")

    # load the image
    image = load_img(image_path)
    i_width, i_height = image.size
    image = img_to_array(image)

    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    plt.imshow(image)
    plt.imsave(report_path+'image.jpg', image)

    # SLICE to tiles (=small tiles = moin tile reference)
    tiles, tile_info, tile_dims = \
        make_tiles_small(
            image, name, i_width, i_height, tile_size, border)
    ## tile_info:
        ### [0:3] tile row, tile column, [true_label], [probability],
        ### [4:5] position horizontal, position vertical  // (top left corner of tile in the image)
        ### [6:7] [vertical label position], [horizontal label] position within the (small) tile

    # EXPAND tiles to large tiles for the model as input
    tiles_large = \
        expand_tiles(tile_info, tile_dims, border, image, tile_size)

    return image, tiles_large, tiles, tile_info


def make_pixels_predicted(
        pos_list, pred_pos_tiles, tile_info, border):

    # tile border must not be removed               (-border)
    # because it is not incl. in tile_info position
    pixels_pred = np.array(pred_pos_tiles)

    # Adjust the predicted coordinates to the original image
    for pixel in range(len(pos_list)):
        tile_num = pos_list[pixel]
        pixels_pred[pixel, 0] += tile_info[tile_num, 4]
        pixels_pred[pixel, 1] += tile_info[tile_num, 5]

    pixels_pred = pixels_pred.astype(int)
    print(f"pixels_pred: {pixels_pred.shape}")

    return pixels_pred, tile_info

