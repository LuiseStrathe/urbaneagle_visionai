"""
These functions load the required raw data.
"""

#------------------------------------------------------------#

from matplotlib import pyplot as plt

#------------------------------------------------------------#



def show_img_with_labels(image=orig_image, labels=labels, marker=','):
    plt.imshow(image)
    # plot each labeled tree
    for dot in labels:
        #print(dot[:,1])
        plt.plot(dot[:,1], dot[:,0], marker=marker, color="green")
    # show the plot
    plt.show()
