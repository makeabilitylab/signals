"""Audio helpers (mono conversion and analysis) built on librosa/scipy for the
Tutorials notebooks.
"""
import matplotlib.pyplot as plt  # matplotlib: https://matplotlib.org/
import numpy as np  # numpy: https://numpy.org/
import scipy as sp # for signal processing
from scipy import signal
from scipy.spatial import distance
import librosa
import random

def convert_to_mono(audio_data):
    '''Converts stereo audio (a 2-D array) to mono by averaging the two channels; mono input is returned unchanged.'''
    if len(audio_data.shape) == 2:
        print("Converting stereo audio file to mono")
        audio_data_mono = audio_data.sum(axis=1) / 2
        return audio_data_mono
    return audio_data