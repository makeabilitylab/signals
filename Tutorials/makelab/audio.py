import matplotlib.pyplot as plt # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import numpy as np # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp # for signal processing
from scipy import signal
from scipy.spatial import distance
import librosa
import random

def convert_to_mono(audio_data):
    if len(audio_data.shape) == 2:
        print("Converting stereo audio file to mono")
        audio_data_mono = audio_data.sum(axis=1) / 2
        return audio_data_mono
    return audio_data