"""
These functions load the required raw data.
"""

# -------------------------IMPORTS-----------------------#

from matplotlib import pyplot as plt


#------------------------FUNCTIONS-----------------------#

def show_true_tiles(tile_info, name_raw_data, path):

    # show true tiles
    num_ver = int(tile_info[:, 0].max()) + 1
    num_hor = int(tile_info[:, 1].max()) + 1
    tiles_show = tile_info[:, 2].reshape(num_ver, num_hor)
    plt.imshow(tiles_show, cmap='gray')
    plt.title(f"Labeled true tiles of image {name_raw_data}\n"
              f"The shape of the tiled picture is {tiles_show.shape}\n"
              f"{tile_info[:, 2].sum()} tiles are labeled as trees.")
    plt.show()
    if path is not None:
        plt.savefig(f'{path}figures/true_tiles.png')
    plt.close()
    pass


def show_pred_tiles(predictions, tile_info, path_model, name_raw_data):

    # show prediction of tiles
    num_ver = int(tile_info[:, 0].max()) + 1
    num_hor = int(tile_info[:, 1].max()) + 1
    tiles_show = predictions.reshape(num_ver, num_hor)
    plt.imshow(tiles_show, cmap='gray')
    if path_model is not None:
        plt.savefig(f'{path_model}tiles.png')
    num_pos = (predictions > 0).sum()
    plt.title(f"Predicted tiles of image {name_raw_data}\n"
              f"{num_pos} out of {len(predictions)} tiles we predicted as with trees.\n"
              f"This is {num_pos / len(predictions)}% of the tiles")
    plt.show()
    print(f"{num_pos} out of {len(predictions)} tiles we predicted as with trees!")
    print(f"This is {round(num_pos / len(predictions)*100,0)}% of the tiles")
    pass


def draw_img_with_pixels(image, pixels, path=None):

    color = [255./255, 1./255, 130./255]

    # draw pixels into image
    for pixel in pixels:
        # central pixel
        ver_pixel = pixel[0]
        hor_pixel = pixel[1]
        #image[ver_pixel, hor_pixel] = color

        # frame
        expand = 9
        for horizontal in range(hor_pixel-expand, hor_pixel+expand):
            image[(ver_pixel-expand), (horizontal)] = color
            image[(ver_pixel+expand), (horizontal)] = color
        for vertical in range(ver_pixel-expand, ver_pixel+expand):
            image[(vertical), (hor_pixel-expand)] = color
            image[(vertical), (hor_pixel+expand)] = color

    # draw full image
    plt.imshow(image)
    plt.title(f"Number of predicted trees: {len(pixels)}")
    if path is not None:
        plt.imsave(path+'image_trees.jpg', image)
    plt.close()

    # draw pixels into image
    for pixel in pixels:
        # central pixel
        ver_pixel = pixel[0]
        hor_pixel = pixel[1]
        #image[ver_pixel, hor_pixel] = color

        # frame
        expand = 8
        for horizontal in range(hor_pixel-expand, hor_pixel+expand):
            image[(ver_pixel-expand):(ver_pixel+expand), (horizontal)] = color

    plt.imshow(image)
    plt.title(f"Number of predicted trees: {len(pixels)}")
    if path is not None:
        plt.imsave(path+'image_trees_bold.jpg', image)
    plt.close()



    pass