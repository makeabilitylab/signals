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

def plot_signal_to_axes(ax, s, sampling_rate, title=None, signal_label=None, marker=None):
    '''Plots a sine wave s with the given sampling rate
    
    Parameters:
    ax: matplot axis to do the plotting
    s: numpy array
    sampling_rate: sampling rate of s
    title: chart title
    signal_label: the label of the signal
    '''
    ax.plot(s, label=signal_label, marker=marker, alpha=0.9)
    ax.set(xlabel="samples")
    
    if signal_label is not None:
        ax.legend()

    # we use y=1.14 to make room for the secondary x-axis
    # see: https://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
    if title is not None:
        ax.set_title(title, y=1.1)
    
    ax.grid()

    # add in a secondary x-axis to draw the x ticks as time (rather than samples)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("time (secs)")
    ax_ticks = ax.get_xticks()[1:-1]
    
    ax2_tick_labels = ax.get_xticks()[1:-1] / sampling_rate
    ax2_tick_labels = ['{:.2f}s'.format(x) for x in ax2_tick_labels]
    ax2.set_xticks(ax_ticks)
    ax2.set_xticklabels(ax2_tick_labels)
    
def plot_signal(s, sampling_rate, quantization_bits = 16, xlim_zoom = None, highlight_zoom_area = True):
    '''Plots audio data with the given sampling_rate, quantization level, and xlim_zoom'''
    
    if xlim_zoom == None:
        fig, axes = plt.subplots(1, 1, figsize=(15,6))
        plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"
        plot_signal_to_axes(axes, s, sampling_rate, plot_title)
        return (fig, axes)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15,6), sharey=True, gridspec_kw={'width_ratios': [2, 1]})
        plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"
        plot_signal_to_axes(axes[0], s, sampling_rate, plot_title)
        
        # if(xlim_zoom == None):
        #     xlim_zoom = get_random_xzoom(len(audio_data), 0.1)
        
        if highlight_zoom_area:
            # yellow highlight color: color='#FFFBCC'
            axes[0].axvspan(xlim_zoom[0], xlim_zoom[1], color='orange', alpha=0.3)
            
        axes[1].set_xlim(xlim_zoom)
        plot_signal_to_axes(axes[1], s, sampling_rate, plot_title + ' zoomed')
        fig.tight_layout()
        return (fig, axes)

def get_random_xzoom(max_length, fraction_of_length):
    '''Returns a tuple of (start, end) for a random xzoom amount''' 
    zoom_length = int(max_length * fraction_of_length)
    random_start = random.randint(0, max_length - zoom_length)
    xlim_zoom = (random_start, random_start + zoom_length)
    return xlim_zoom