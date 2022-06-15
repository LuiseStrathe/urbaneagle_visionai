"""
These functions load and transform data for tree identification.
"""

#----------------------IMPORTS---------------------------#

import os


#------------------------FUNCTIONS-----------------------#

def make_directories(path):
    if not os.path.exists(f"{path}"):
        os.mkdir(f"{path}")
    if not os.path.exists(f"{path}/figures"):
        os.mkdir(f"{path}/figures")

    if not os.path.exists(f"{path}/images"):
        os.mkdir(f"{path}/images")