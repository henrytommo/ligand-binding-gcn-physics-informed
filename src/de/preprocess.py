# pre processing data files from pdbbind+

import pandas as pd
import glob

DATA_FOLDER = "data"
INDEX_FILE_PL = "INDEX_general_PL.2020R1.lst"
PL_FOLDER = "P-L/*"


def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def load_index_files(index_file):
    index_list = []
    with open(index_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                pass
            else:
                parts = [
                    l.strip()
                    for lst in [p.split(",") for p in line.split("  ") if p != ""]
                    for l in lst
                ]
                index_list.append(parts)
    return index_list


def get_thermo_descriptor(thermo_descriptor, index_list):
    """
    Returns a list of lists, with each sublist containing the pdb indicator
    and the thermo_descriptor (Ki=, Kd=, etc).
    To remove the free text description, cut off at len 12 - review this
    """
    thermo_list = []
    for lst in index_list:
        idx = index_containing_substring(lst, thermo_descriptor)
        if idx != -1 and len(lst[idx]) < 12:
            thermo_list.append([lst[0], lst[idx]])
