"""
These functions contain small helper features.
"""

#----------------------IMPORTS---------------------------#

import os
from pathlib import Path
from resource import getpagesize


#------------------------FUNCTIONS-----------------------#

def make_directories(path):
    if not os.path.exists(f"{path}"):
        os.mkdir(f"{path}")
    if not os.path.exists(f"{path}/figures"):
        os.mkdir(f"{path}/figures")

    if not os.path.exists(f"{path}/images"):
        os.mkdir(f"{path}/images")


def get_resident_set_size() -> int:
    """Return the current resident set size in bytes."""
    # statm columns are: size resident shared text lib data dt
    statm = Path('/proc/self/statm').read_text()
    fields = statm.split()
    return int(fields[1]) * getpagesize()
# use with this:
data = []
start_memory = get_resident_set_size()
for _ in range(4):
    data.append('X' * 100000)
    print(get_resident_set_size() - start_memory)

