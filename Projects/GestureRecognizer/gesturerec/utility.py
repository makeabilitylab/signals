"""File-handling helpers for loading gesture logs.

`find_csv_filenames` deliberately excludes any `fulldatastream` files so that
per-trial CSV loading and full-stream loading don't collide.
"""
from os import listdir
import ntpath
import os


def find_csv_filenames(path_to_dir, suffix=".csv"):
    '''Returns all CSV filenames in path_to_dir, excluding any containing "fulldatastream".'''
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames
            if filename.endswith(suffix) and "fulldatastream" not in filename]


def path_leaf(path):
    '''Returns the final component (leaf) of a path. From https://stackoverflow.com/a/8384788'''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_immediate_subdirectories(a_dir):
    '''Returns the names of the immediate subdirectories of a_dir.'''
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def extract_gesture_name(filename):
    '''Extracts the gesture name from a log filename (the text before the first "_").'''
    token_split_pos = filename.index('_')
    return filename[:token_split_pos]
