import matplotlib.pyplot as plt # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import numpy as np # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp # for signal processing
from scipy import signal
from scipy.spatial import distance
import librosa
import random

### SINE AND COSINE GENERATOR FUNCTIONS ###

def create_sine_waves(freqs, sampling_rate, total_time_in_secs = None, return_time = False):
    '''Creates multiple sine waves corresponding to the freq array, sampling rate, and length
    
       Returns a tuple list of (freq, sine_wave) or (freq, (time, sine_wave))
       depending on whether return_time is True or False
    '''
    sine_waves = []
    for freq in freqs:
        sine_waves.append((freq, create_sine_wave(freq, sampling_rate, total_time_in_secs, return_time)))
    return sine_waves

def create_sine_wave_sequence(freqs, sampling_rate, time_per_freq = None, starting_amplitudes = None,
                             ending_amplitudes = None):

    '''
    Creates a sine wave sequence at the given frequencies and sampling rate. You can control
    the time per frequency via time_per_freq and the starting and ending amplitudes of each signal
    
    Parameters:
        freqs (array): array of frequencies
        sampling_rate (num): sampling rate
        time_per_freq (float or list): If a float, creates all sine waves of that length (in secs).
            If an array, takes the time per frequency (in secs). If None, sets all sine waves to length 1 sec.
        starting_amplitudes (array): List of starting amplitudes for each freq (one by default)
        ending_amplitudes (array): list of ending amplitudes for each freq (zero by default)
    '''
    if starting_amplitudes is None:
        starting_amplitudes = np.ones(len(freqs))

    if ending_amplitudes is None:
        ending_amplitudes = np.zeros(len(freqs))

    if time_per_freq is None:
        time_per_freq = np.ones(len(freqs))
    elif isinstance(time_per_freq, (list, tuple, np.ndarray)) is False:
        time_per_freq = np.full(len(freqs), time_per_freq)

    signal_sequence = np.array([])
    for i, freq in enumerate(freqs):
        signal = create_sine_wave(freq, sampling_rate, time_per_freq[i])
        
        # Currently linear interpolation between start and end amplitudes
        # But we could expand this later to exponential, etc.
        amplitudes = np.linspace(starting_amplitudes[i], ending_amplitudes[i], num=len(signal))
        signal_with_amplitudes = signal * amplitudes
        signal_sequence = np.concatenate((signal_sequence, signal_with_amplitudes))
    
    return signal_sequence

def create_composite_sine_wave(freqs, sampling_rate, total_time_in_secs, amplitudes = None,
                              use_random_amplitudes = False, return_time = False):
    '''Creates a composite sine wave with the given frequencies and amplitudes'''
    
    if amplitudes is None and use_random_amplitudes is False:
        amplitudes = np.ones(len(freqs))
    elif amplitudes is None and use_random_amplitudes is True:
        amplitudes = np.random.uniform(low = 0.1, high = 1, size=(len(freqs)))

    time = np.arange(total_time_in_secs * sampling_rate) / sampling_rate
    signal_composite = np.zeros(len(time)) # start with empty array
    for i, freq in enumerate(freqs):
        # set random amplitude for each freq (you can change this, of course)
        signal = amplitudes[i] * create_sine_wave(freq, sampling_rate, total_time_in_secs)
        signal_composite += signal

    if return_time is False:
        return signal_composite
    else:
        return (time, signal_composite)

def create_sine_wave(freq, sampling_rate, total_time_in_secs = None, return_time = False):
    '''Creates a sine wave with the given frequency, sampling rate, and length'''
    
    # if the total time in secs is None, then return one period of the wave
    if total_time_in_secs is None:
        total_time_in_secs = 1 / freq

    # Create an array from 0 to total_time_in_secs * sampling_rate (and then divide by sampling
    # rate to get each time_step)
    time = np.arange(total_time_in_secs * sampling_rate) / sampling_rate
    
    # Could also generate this signal by:
    # time = np.linspace(0, total_time_in_secs, int(total_time_in_secs * sampling_rate), endpoint=False)

    sine_wave = np.sin(2 * np.pi * freq * time)

    # or, once the sample is made:
    # time = np.linspace(0, len(s) / sampling_rate, num=len(s))

    if return_time is False:
        return sine_wave
    else:
        return (time, sine_wave)

def create_cos_wave(freq, sampling_rate, total_time_in_secs = None, return_time = False):
    '''Creates a cos wave with the given frequency, sampling rate, and length'''
    
     # if the total time in secs is None, then return one period of the wave
    if total_time_in_secs is None:
        total_time_in_secs = 1 / freq

    # Create an array from 0 to total_time_in_secs * sampling_rate (and then divide by sampling
    # rate to get each time_step)
    time = np.arange(total_time_in_secs * sampling_rate) / sampling_rate
    cos_wave = np.cos(2 * np.pi * freq * time)

    if return_time is False:
        return cos_wave
    else:
        return (time, cos_wave)

def get_random_xzoom(signal_length, fraction_of_length):
    '''Returns a tuple of (start, end) for a random xzoom amount''' 
    zoom_length = int(signal_length * fraction_of_length)
    random_start = random.randint(0, signal_length - zoom_length)
    xlim_zoom = (random_start, random_start + zoom_length)
    return xlim_zoom

def map(val, start1, stop1, start2, stop2):
    '''Similar to Processing and Arduino's map function'''
    return ((val-start1)/(stop1-start1)) * (stop2 - start2) + start2

def remap(val, start1, stop1, start2, stop2):
    '''Similar to Processing and Arduino's map function'''
    return ((val-start1)/(stop1-start1)) * (stop2 - start2) + start2

### SIGNAL MANIPULATION FUNCTIONS ###

# While numpy provides a roll function, it does not appear to provide a shift
# https://stackoverflow.com/q/30399534
# So, lots of people have implemented their own, including some nice benchmarks here:
# https://stackoverflow.com/a/42642326
def shift_array(arr, shift_amount, fill_value = np.nan):
    '''Shifts the array either left or right by the shift_amount (which can be negative or positive)
     
       From: https://stackoverflow.com/a/42642326
    '''
    result = np.empty_like(arr)
    if shift_amount > 0:
        result[:shift_amount] = fill_value
        result[shift_amount:] = arr[:-shift_amount]
    elif shift_amount < 0:
        result[shift_amount:] = fill_value
        result[:shift_amount] = arr[-shift_amount:]
    else:
        result[:] = arr
    return result

### SIGNAL ANALYSIS FUNCTIONS ###

# TODO: update get_top_n_frequency_indices_sorted so that you can specify a min_gap
#       between top freqs (so if two top freqs are close together, one can be skipped)
def get_top_n_frequency_indices_sorted(n, freqs, amplitudes):
    '''Gets the top N frequency indices (sorted)'''
    ind = np.argpartition(amplitudes, -n)[-n:] # from https://stackoverflow.com/a/23734295
    ind_sorted_by_coef = ind[np.argsort(-amplitudes[ind])] # reverse sort indices

    return ind_sorted_by_coef

def calc_zero_crossings(s, min_gap = None):
    '''Returns the number of zero crossings in the signal s
    
    This method is based on https://stackoverflow.com/q/3843017
    
    Parameters:
    s: the signal
    min_gap: the minimum gap (in samples) between zero crossings
    TODO:     
    - could have a mininum height after the zero crossing (within some window) to eliminate noise
    '''
    # I could not get the speedier Pythonista solutions to work reliably so here's a 
    # custom non-Pythony solution
    cur_pt = s[0]
    zero_crossings = []
    last_zero_crossing_idx = None
    last_zero_cross_idx_saved = None
    for i in range(1, len(s)):
        next_pt = s[i]
        zero_crossing_idx = None
        
        # There are three cases to check for:
        #  1. If the cur_pt is gt zero and the next_pt is lt zero, obviously a zero crossing.
        #     Similarly, if the next_pt is gt zero and the cut_pt is lt zero, again a zero crossing
        #  2. If the cur_pt is zero and the next_pt gt zero, then we walk back to see when zero 
        #     was first "entered"
        #  3. Finally, if the cut_pt is zero and the next_pt lt zero, we again walk back to see
        #     when zero was first "entered"
        if ((next_pt < 0 and cur_pt > 0) or (next_pt > 0 and cur_pt < 0)):
            # if we're here, a zero crossing occurred
            zero_crossing_idx = i
          
        elif cur_pt == 0 and next_pt > 0:
            # check for previous points less than 0
            # as soon as tmp_pt is not zero, we are done
            tmp_pt = cur_pt
            walk_back_idx = i
            while(tmp_pt == 0 and walk_back_idx > 0):
                walk_back_idx -= 1
                tmp_pt = s[walk_back_idx]
            
            if tmp_pt < 0:
                zero_crossing_idx = i
                
        elif cur_pt == 0 and next_pt < 0:
            # check for previous points greater than 0
            # as soon as tmp_pt is not zero, we are done
            tmp_pt = cur_pt
            walk_back_idx = i
            while(tmp_pt == 0 and walk_back_idx > 0):
                walk_back_idx -= 1
                tmp_pt = s[walk_back_idx]
            
            if tmp_pt > 0:
                zero_crossing_idx = i
        
        # now potentially add zero_crossing_idx to our list
        if zero_crossing_idx is not None:
            # potentially have a new zero crossing, check for other conditions
            if last_zero_cross_idx_saved is None or \
               last_zero_cross_idx_saved is not None and min_gap is None or \
               (min_gap is not None and (i - last_zero_cross_idx_saved) > min_gap):
                
                zero_crossings.append(zero_crossing_idx) # save the zero crossing point
                last_zero_cross_idx_saved = zero_crossing_idx
            
            last_zero_crossing_idx = zero_crossing_idx
            
        cur_pt = s[i]
    return zero_crossings

##### VISUALIZATION CODE ######
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
    ax.set(xlabel="Samples")
    ax.set(ylabel="Amplitude")
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
    
    ax_ticks = ax.get_xticks()[1:-1]
    ax2_tick_labels = ax.get_xticks()[1:-1] / sampling_rate

    num_samples_shown = ax.get_xlim()[1] - ax.get_xlim()[0]
    time_shown = num_samples_shown / sampling_rate
    if time_shown < 1:
        ax2.set_xlabel("Time (ms)")
        # format with 'g' causes insignificant trailing zeroes to be removed
        # https://stackoverflow.com/a/2440708 but also uses scientific notation, oh well!
        ax2_tick_labels = [f"{x * 1000:.1f}" for x in ax2_tick_labels]
    else:
        ax2.set_xlabel("Time (secs)")
        ax2_tick_labels = ['{:.2f}'.format(x) for x in ax2_tick_labels]

    ax2.set_xticks(ax_ticks)
    ax2.set_xticklabels(ax2_tick_labels)

def plot_audio(s, sampling_rate, quantization_bits = 16, title = None, xlim_zoom = None, highlight_zoom_area = True):
    ''' Calls plot_Signal but accepts quantization_bits '''
    plot_title = title
    if plot_title is None:
        plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"
    
    return plot_signal(s, sampling_rate, title = title, xlim_zoom = xlim_zoom, highlight_zoom_area = highlight_zoom_area)

def plot_signal(s, sampling_rate, title = None, xlim_zoom = None, highlight_zoom_area = True):
    '''Plots time-series data with the given sampling_rate and xlim_zoom'''
    
    plot_title = title
    if plot_title is None:
        plot_title = f"Sampling rate: {sampling_rate} Hz"

    if xlim_zoom == None:
        fig, axes = plt.subplots(1, 1, figsize=(15,6))
        
        plot_signal_to_axes(axes, s, sampling_rate, plot_title)
        return (fig, axes)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15,6), sharey=True, gridspec_kw={'width_ratios': [2, 1]})
        plot_signal_to_axes(axes[0], s, sampling_rate, plot_title)
        
        # if(xlim_zoom == None):
        #     xlim_zoom = get_random_xzoom(len(audio_data), 0.1)
        
        if highlight_zoom_area:
            # yellow highlight color: color='#FFFBCC'
            axes[0].axvspan(xlim_zoom[0], xlim_zoom[1], color='orange', alpha=0.3)
            
        axes[1].set_xlim(xlim_zoom)
        zoom_title = f"Signal zoomed: {int(xlim_zoom[0])} - {int(xlim_zoom[1])} samples"
        plot_signal_to_axes(axes[1], s, sampling_rate, zoom_title)
        axes[1].set_ylabel(None)
        fig.tight_layout()
        return (fig, axes)

def plot_sampling_demonstration(total_time_in_secs, real_world_freqs, real_world_continuous_speed = 10000, resample_factor = 200):
    '''Used to demonstrate digital sampling and uses stem plots to show where samples taken'''
    num_charts = len(real_world_freqs)
    fig_height = num_charts * 3.25
    fig, axes = plt.subplots(num_charts, 1, figsize=(15, fig_height))
    
    time = None
    
    i = 0
    sampling_rate = real_world_continuous_speed / resample_factor
    print(f"Sampling rate: {sampling_rate} Hz")
    for real_world_freq in real_world_freqs:
        time, real_world_signal = create_sine_wave(real_world_freq, real_world_continuous_speed, 
                                               total_time_in_secs, return_time = True)
        sampled_time = time[::resample_factor]
        sampled_signal = real_world_signal[::resample_factor]
        
        axes[i].plot(time, real_world_signal)
        axes[i].axhline(0, color="gray", linestyle="-", linewidth=0.5)
        axes[i].plot(sampled_time, sampled_signal, linestyle='None', alpha=0.8, marker='s', color='black')
        axes[i].vlines(sampled_time, ymin=0, ymax=sampled_signal, linestyle='-.', alpha=0.8, color='black')
        axes[i].set_ylabel("Amplitude")
        axes[i].set_xlabel("Time (secs)")
        axes[i].set_title(f"{real_world_freq}Hz signal sampled at {sampling_rate}Hz")
        
        i += 1
    fig.tight_layout(pad = 3.0)

#### FREQUENCY VISUALIZATIONS ####
def plot_signal_and_magnitude_spectrum(t, s, sampling_rate, title = None, xlim_zoom_in_secs = None):
    # Plot the time domain
    ax_main_time = None
    ax_zoom_time = None
    ax_spectrum = None
    
    axes = []
    if xlim_zoom_in_secs is None:
        fig, axes = plt.subplots(2, 1, figsize=(15,8))
        ax_main_time = axes[0]
        ax_spectrum = axes[1]
    else:
        fig = plt.figure(figsize=(15, 9))

        # TODO: in future, ideally, we would have like a 70/30 split on top row
        ax_main_time = plt.subplot2grid((2, 2), (0, 0))
        ax_zoom_time = plt.subplot2grid((2, 2), (0, 1))
        ax_spectrum = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        axes.append(ax_main_time)
        axes.append(ax_zoom_time)
        axes.append(ax_spectrum)
    
    # Plot main time domain
    ax_main_time.plot(t, s)
    
    if title is None:
        title = "Signal (time domain)"
    
    ax_main_time.set_title(title)
    ax_main_time.set_ylabel("Amplitude")
    ax_main_time.set_xlabel("Time (secs)")
    
    if ax_zoom_time is not None:
        # plot zoom
        ax_main_time.axvspan(xlim_zoom_in_secs[0], xlim_zoom_in_secs[1], color='orange', alpha=0.3)
        
        ax_zoom_time.set_xlim(xlim_zoom_in_secs)
        ax_zoom_time.plot(t, s)  
        ax_zoom_time.set_title(title + " (Zoomed)")
        ax_zoom_time.set_ylabel("Amplitude")
        ax_zoom_time.set_xlabel("Time (secs)")

    # Plot the frequency transform
    ax_spectrum.magnitude_spectrum(s, Fs = sampling_rate, color='r')
    fig.tight_layout()

    return (fig, axes)

import matplotlib.ticker as ticker
def plot_spectrogram_to_axes(ax, s, sampling_rate, title=None, 
                             marker=None, custom_axes = True):
    '''Plots a spectrogram wave s with the given sampling rate
    
    Parameters:
    ax: matplot axis to do the plotting
    s: numpy array
    sampling_rate: sampling rate of s
    title: chart title
    '''

    specgram_return_data = ax.specgram(s, Fs=sampling_rate)

    # we use y=1.14 to make room for the secondary x-axis
    # see: https://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
    if title is not None:
        ax.set_title(title, y=1.2)

    ax.set_ylabel("Frequency")

    # add in a secondary x-axis to draw the x ticks as time (rather than samples)
    if custom_axes:
        ax.set(xlabel="Samples")
        ax_xtick_labels = np.array(ax.get_xticks()) * sampling_rate
        ax_xtick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
        ax.set_xticklabels(ax_xtick_labels_strs)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel("Time (secs)")
        ax2_tick_labels = ax_xtick_labels / sampling_rate
        ax2_tick_labels_strs = [f"{xtick_label:.1f}s" for xtick_label in ax2_tick_labels]
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(ax2_tick_labels_strs)
    return specgram_return_data
    
def plot_spectrogram(s, sampling_rate, title = None, xlim_zoom = None, highlight_zoom_area = True):
    '''Plots signal with the given sampling_Rate, quantization level, and xlim_zoom'''
    fig, axes = plt.subplots(1, 2, figsize=(15,4), gridspec_kw={'width_ratios': [2, 1]})
    
    if title is None:
        title = f"{len(s) * sampling_rate} sec Signal with {sampling_rate} Hz"
    
    specgram_return_data0 = plot_spectrogram_to_axes(axes[0], s, sampling_rate, title)
    
    if(xlim_zoom == None):
        max_length = len(s)
        length = int(max_length * 0.1)
        random_start = random.randint(0, max_length - length)
        xlim_zoom = (random_start, random_start + length)
      
    axes[1].set_xlim(xlim_zoom)
    #axes[1].set_xlim(12000, 14000)
    specgram_return_data1 = plot_spectrogram_to_axes(axes[1], s, sampling_rate, title + ' (Zoomed)', custom_axes = False)
    
    zoom_x1 = xlim_zoom[0] / sampling_rate
    zoom_x2 = xlim_zoom[1] / sampling_rate
    axes[1].set_xlim(zoom_x1, zoom_x2) # but this one seems to work
    
    ax2 = axes[1].twiny()
    ax2.set_xlim(axes[1].get_xlim())
    ax2.set_xticks(axes[1].get_xticks())
    ax2_tick_labels_strs = [f"{xtick_label:.1f}s" for xtick_label in axes[1].get_xticks()]
    ax2.set_xticklabels(ax2_tick_labels_strs)
    ax2.set_xlabel("Time (secs)")
    
    ax_xtick_labels = np.array(axes[1].get_xticks()) * sampling_rate
    ax2_tick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
    axes[1].set(xlabel="Samples")
    axes[1].set_xticklabels(ax2_tick_labels_strs)
    
    if highlight_zoom_area:
        # yellow highlight color: color='#FFFBCC'
        axes[0].axvline(x = zoom_x1, linewidth=2, color='r', alpha=0.8, linestyle='-.')
        axes[0].axvline(x = zoom_x2, linewidth=2, color='r', alpha=0.8, linestyle='-.')
    
    fig.tight_layout()
    return (fig, axes, specgram_return_data0, specgram_return_data1)

def plot_signal_and_spectrogram(s, sampling_rate, quantization_bits, xlim_zoom = None, highlight_zoom_area = True):
    '''Plot waveforms and spectrograms together'''
    fig = plt.figure(figsize=(15, 9))
    spec = fig.add_gridspec(ncols = 2, nrows = 2, width_ratios = [2, 1], height_ratios = [1, 1])
    plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"
    
    ax_waveform1 = plt.subplot(spec[0, 0])
    ax_waveform1.set_xlim(0, len(s))
    ax_waveform2 = plt.subplot(spec[0, 1], sharey = ax_waveform1)

    ax_spectrogram1 = plt.subplot(spec[1, 0])
    ax_spectrogram2 = plt.subplot(spec[1, 1])

    plot_signal_to_axes(ax_waveform1, s, sampling_rate, plot_title)
    specgram_return_data = plot_spectrogram_to_axes(ax_spectrogram1, s, sampling_rate, plot_title)
    #print(len(specgram_return_data[2]))
    
    #print(ax_waveform1.get_xlim())
    #print(ax_spectrogram1.get_xlim())
    waveform_xrange = ax_waveform1.get_xlim()[1] - ax_waveform1.get_xlim()[0]

    ax_waveform2.set_xlim(xlim_zoom)
    plot_signal_to_axes(ax_waveform2, s, sampling_rate, plot_title + ' zoomed')
    
    zoom_x1 = remap(xlim_zoom[0], ax_waveform1.get_xlim()[0], ax_waveform1.get_xlim()[1], 
                    ax_spectrogram1.get_xlim()[0], ax_spectrogram1.get_xlim()[1])
    zoom_x2 = remap(xlim_zoom[1], ax_waveform1.get_xlim()[0], ax_waveform1.get_xlim()[1], 
                    ax_spectrogram1.get_xlim()[0], ax_spectrogram1.get_xlim()[1])
    
    #print(ax_spectrogram2.get_xlim(), zoom_x1, zoom_x2)
    ax_spectrogram2.set_xlim(zoom_x1, zoom_x2) # this won't make a difference
    plot_spectrogram_to_axes(ax_spectrogram2, s, sampling_rate, plot_title, 
                             custom_axes = False)
    ax_spectrogram2.set_xlim(zoom_x1, zoom_x2) # but this one seems to work
     
    ax2 = ax_spectrogram2.twiny()
    ax2.set_xlim(ax_spectrogram2.get_xlim())
    ax2.set_xticks(ax_spectrogram2.get_xticks())
    ax2_tick_labels_strs = [f"{xtick_label:.2f}s" for xtick_label in ax_spectrogram2.get_xticks()]
    ax2.set_xticklabels(ax2_tick_labels_strs)
    ax2.set_xlabel("Time (secs)")
    
    ax_xtick_labels = np.array(ax_spectrogram2.get_xticks()) * sampling_rate
    ax2_tick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
    ax_spectrogram2.set(xlabel="Samples")
    ax_spectrogram2.set_xticks(ax_spectrogram2.get_xticks())
    ax_spectrogram2.set_xticklabels(ax2_tick_labels_strs)
    
    if highlight_zoom_area:
        # yellow highlight color: color='#FFFBCC'
        ax_waveform1.axvspan(xlim_zoom[0], xlim_zoom[1], color='orange', alpha=0.3)
        ax_spectrogram1.axvline(x = zoom_x1, linewidth=2, color='r', alpha=0.8, linestyle='-.')
        ax_spectrogram1.axvline(x = zoom_x2, linewidth=2, color='r', alpha=0.8, linestyle='-.')
    
    fig.tight_layout()