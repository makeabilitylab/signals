"""Signal generation, analysis, and plotting helpers for the Tutorials notebooks.

This module is written to be *read by students*: the implementations favor clarity
over cleverness, cite their sources, and often show the alternative ways a result
could be computed. Where a function duplicates something a production library already
does, a comment points to that library equivalent.
"""
import random

import matplotlib.pyplot as plt  # the premier plotting library for Python: https://matplotlib.org/
import matplotlib.ticker as ticker  # for FixedLocator/FixedFormatter when setting custom ticks
import numpy as np  # the premier numerical/signal-handling library for Python: https://numpy.org/

### SINE AND COSINE GENERATOR FUNCTIONS ###

def create_sine_waves(freqs, sampling_rate, total_time_in_secs = None, return_time = False):
    '''Creates one sine wave per frequency in freqs (see create_sine_wave for the details).

    Parameters:
        freqs (array): the frequencies (in Hz) to generate, one wave each
        sampling_rate (num): samples per second
        total_time_in_secs (float): length of each wave in secs (defaults to one period)
        return_time (bool): if True, include each wave's time array

    Returns:
        list: a list of (freq, sine_wave) tuples, or (freq, (time, sine_wave)) tuples
            when return_time is True.
    '''
    sine_waves = []
    for freq in freqs:
        sine_waves.append((freq, create_sine_wave(freq, sampling_rate, total_time_in_secs, return_time)))
    return sine_waves

def create_sine_wave_sequence(freqs, sampling_rate, time_per_freq = None, starting_amplitudes = None,
                             ending_amplitudes = None):

    '''Creates a sequence of sine waves played back-to-back (concatenated), one per frequency.

    Each wave's amplitude is linearly ramped from its starting to its ending value, which lets
    you fade notes in/out so the joins between them don't "click".

    Parameters:
        freqs (array): array of frequencies (in Hz)
        sampling_rate (num): samples per second
        time_per_freq (float or list): If a float, every wave gets that length (in secs).
            If a list/array, the per-frequency length (in secs). If None, every wave is 1 sec.
        starting_amplitudes (array): starting amplitude for each freq (one by default)
        ending_amplitudes (array): ending amplitude for each freq (zero by default)

    Returns:
        np.ndarray: the concatenated 1-D signal.
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
    '''Creates a single composite signal by summing sine waves at the given frequencies.

    This is how a chord (or any complex tone) is built: add together pure sine waves. The FFT
    notebooks then pull those component frequencies back out.

    Parameters:
        freqs (array): the component frequencies (in Hz) to sum
        sampling_rate (num): samples per second
        total_time_in_secs (float): length of the signal in secs
        amplitudes (array): amplitude per freq. Defaults to all ones, unless use_random_amplitudes.
        use_random_amplitudes (bool): if True (and amplitudes is None), pick random amplitudes in [0.1, 1)
        return_time (bool): if True, also return the time array

    Returns:
        np.ndarray, or (time, np.ndarray) if return_time is True.
    '''

    if amplitudes is None and use_random_amplitudes is False:
        amplitudes = np.ones(len(freqs))
    elif amplitudes is None and use_random_amplitudes is True:
        amplitudes = np.random.uniform(low = 0.1, high = 1, size=(len(freqs)))

    time = np.arange(total_time_in_secs * sampling_rate) / sampling_rate
    signal_composite = np.zeros(len(time)) # start with a flat (all-zeros) signal
    for i, freq in enumerate(freqs):
        # scale this component by its amplitude, then add it into the running sum
        signal = amplitudes[i] * create_sine_wave(freq, sampling_rate, total_time_in_secs)
        signal_composite += signal

    if return_time is False:
        return signal_composite
    else:
        return (time, signal_composite)

def create_sine_wave(freq, sampling_rate, total_time_in_secs = None, return_time = False):
    '''Creates a sine wave with the given frequency, sampling rate, and length.

    Parameters:
        freq (num): frequency in Hz
        sampling_rate (num): samples per second
        total_time_in_secs (float): length in secs. If None, returns exactly one period (1/freq secs).
        return_time (bool): if True, also return the time array

    Returns:
        np.ndarray, or (time, np.ndarray) if return_time is True.
    '''

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
    '''Creates a cosine wave with the given frequency, sampling rate, and length.

    Same contract as create_sine_wave, but cosine (so it starts at amplitude 1, not 0).

    Returns:
        np.ndarray, or (time, np.ndarray) if return_time is True.
    '''

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
    '''Returns a random (start, end) sample range covering fraction_of_length of the signal.'''
    zoom_length = int(signal_length * fraction_of_length)
    random_start = random.randint(0, signal_length - zoom_length)
    xlim_zoom = (random_start, random_start + zoom_length)
    return xlim_zoom

def remap(val, start1, stop1, start2, stop2):
    '''Linearly re-maps val from the range [start1, stop1] into the range [start2, stop2].

    Like Processing's and Arduino's map() function. numpy offers a close equivalent in
    np.interp(val, [start1, stop1], [start2, stop2]) -- but np.interp *clamps* values that fall
    outside the input range, whereas this version extrapolates (the Arduino behavior students expect).
    '''
    return ((val-start1)/(stop1-start1)) * (stop2 - start2) + start2

# `map` is kept as an alias of `remap` for the Processing/Arduino name students know. Note it
# shadows Python's built-in map() *within this module* -- harmless here since we never use the
# built-in, but it's why notebooks call it as makelab.signal.map(...) rather than bare map(...).
def map(val, start1, stop1, start2, stop2):
    '''Alias of remap() -- see remap for details.'''
    return remap(val, start1, stop1, start2, stop2)

### SIGNAL MANIPULATION FUNCTIONS ###

# While numpy provides a roll function, it does not appear to provide a shift
# https://stackoverflow.com/q/30399534
# So, lots of people have implemented their own, including some nice benchmarks here:
# https://stackoverflow.com/a/42642326
def shift_array(arr, shift_amount, fill_value = np.nan):
    '''Shifts arr left or right by shift_amount samples, filling the vacated end with fill_value.

    A positive shift_amount moves elements to the right (fills the front); a negative shift_amount
    moves them left (fills the back); zero returns a copy unchanged.

    Parameters:
        arr (np.ndarray): the array to shift
        shift_amount (int): samples to shift (sign sets direction)
        fill_value: value placed in the vacated positions (np.nan by default)

    Returns:
        np.ndarray: a new, shifted array (arr is not modified).

    From: https://stackoverflow.com/a/42642326. (scipy offers scipy.ndimage.shift(arr, n, order=0,
    cval=fill_value) for a similar effect, but its default spline interpolation and edge handling
    differ -- this plain index-copy version is clearer for teaching.)
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
    '''Returns the indices of the n largest amplitudes, ordered largest-first.

    Parameters:
        n (int): how many indices to return
        freqs (array): the frequency for each bin (unused here, kept for a symmetric call signature)
        amplitudes (array): the amplitude/magnitude of each bin

    Returns:
        np.ndarray: n indices into `amplitudes`, sorted by descending amplitude.
    '''
    # argpartition finds the n largest in O(len) without fully sorting (from
    # https://stackoverflow.com/a/23734295); we then sort just those n by descending amplitude.
    ind = np.argpartition(amplitudes, -n)[-n:]
    ind_sorted_by_coef = ind[np.argsort(-amplitudes[ind])] # negate to sort high -> low

    return ind_sorted_by_coef

def calc_zero_crossings(s, min_gap = None):
    '''Returns the sample indices where the signal s crosses zero.

    This method is based on https://stackoverflow.com/q/3843017

    Parameters:
        s: the signal
        min_gap: if set, the minimum gap (in samples) required between reported crossings; closer
            crossings are skipped (useful for thinning out noise-driven crossings)

    Returns:
        list: the sample indices of the zero crossings.

    Note: librosa.zero_crossings(s) and np.where(np.diff(np.signbit(s)))[0] both find crossings
    fast, but neither offers the min_gap thinning or the exact-zero "walk back" handling below,
    which is why this is hand-written.

    TODO: could also require a minimum height after the crossing (within some window) to ignore noise.
    '''
    # I could not get the speedier Pythonista solutions to work reliably so here's a
    # custom, step-by-step solution that's easy to follow.
    cur_pt = s[0]
    zero_crossings = []
    last_zero_cross_idx_saved = None  # index of the most recently *saved* crossing (for min_gap)
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
        
        # now potentially add zero_crossing_idx to our list. We save it if this is the first
        # crossing, if no min_gap was requested, or if it's far enough from the last saved one.
        if zero_crossing_idx is not None:
            if last_zero_cross_idx_saved is None or min_gap is None or \
               (i - last_zero_cross_idx_saved) > min_gap:

                zero_crossings.append(zero_crossing_idx) # save the zero crossing point
                last_zero_cross_idx_saved = zero_crossing_idx

        cur_pt = s[i]
    return zero_crossings

##### VISUALIZATION CODE ######
def plot_signal_to_axes(ax, s, sampling_rate, title=None, signal_label=None, marker=None):
    '''Plots time-series signal s onto the given axes, with a second time-based x-axis on top.

    Parameters:
        ax: the matplotlib axes to plot onto
        s: the signal (numpy array)
        sampling_rate: sampling rate of s, used to label the top axis in seconds/ms
        title: chart title (optional)
        signal_label: legend label for the signal (optional)
        marker: matplotlib marker for the data points (optional)
    '''
    ax.plot(s, label=signal_label, marker=marker, alpha=0.9)
    ax.set(xlabel="Samples")
    ax.set(ylabel="Amplitude")
    if signal_label is not None:
        ax.legend()

    # nudge the title up (y=1.1) to make room for the secondary x-axis added below
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
    '''Like plot_signal, but builds a default title that includes the audio's bit depth.

    Returns:
        (fig, axes) from plot_signal.
    '''
    plot_title = title
    if plot_title is None:
        plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"

    # pass plot_title (not the raw title) so the bit-depth default actually reaches the plot
    return plot_signal(s, sampling_rate, title = plot_title, xlim_zoom = xlim_zoom, highlight_zoom_area = highlight_zoom_area)

def plot_signal(s, sampling_rate, title = None, xlim_zoom = None, highlight_zoom_area = True):
    '''Plots time-series data; if xlim_zoom is given, adds a second zoomed-in panel.

    Parameters:
        s: the signal (numpy array)
        sampling_rate: samples per second
        title: chart title (a sampling-rate default is used if None)
        xlim_zoom: optional (start_sample, end_sample) range to show zoomed alongside the full view
        highlight_zoom_area: if True, shade the zoomed range on the full-view panel

    Returns:
        (fig, axes) -- axes is a single Axes when xlim_zoom is None, else a 2-element array.
    '''

    plot_title = title
    if plot_title is None:
        plot_title = f"Sampling rate: {sampling_rate} Hz"

    if xlim_zoom is None:
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
    '''Demonstrates digital sampling: one chart per frequency, with stem markers at the sample points.

    A near-"continuous" signal is generated at real_world_continuous_speed, then sampled every
    resample_factor-th point to show what a real ADC captures. The effective sampling rate is
    real_world_continuous_speed / resample_factor.

    Parameters:
        total_time_in_secs (float): length of each signal in secs
        real_world_freqs (array): one underlying frequency (in Hz) per chart
        real_world_continuous_speed (num): high sampling rate used to approximate the continuous signal
        resample_factor (int): keep every resample_factor-th sample as the "digital" sample
    '''
    num_charts = len(real_world_freqs)
    fig_height = num_charts * 3.25
    fig, axes = plt.subplots(num_charts, 1, figsize=(15, fig_height))

    sampling_rate = real_world_continuous_speed / resample_factor
    print(f"Sampling rate: {sampling_rate} Hz")
    for i, real_world_freq in enumerate(real_world_freqs):
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
    fig.tight_layout(pad = 3.0)

#### FREQUENCY VISUALIZATIONS ####
def plot_signal_and_magnitude_spectrum(t, s, sampling_rate, title = None, xlim_zoom_in_secs = None):
    '''Plots a signal in the time domain alongside its magnitude (frequency) spectrum.

    Parameters:
        t: the time array for s
        s: the signal
        sampling_rate: samples per second (passed to magnitude_spectrum as Fs)
        title: title for the time-domain plot(s)
        xlim_zoom_in_secs: optional (start_sec, end_sec) range; if given, adds a zoomed time panel

    Returns:
        (fig, axes) -- axes is [main_time, spectrum] without zoom, else [main_time, zoom_time, spectrum].
    '''
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

def plot_spectrogram_to_axes(ax, s, sampling_rate, title=None,
                             marker=None, custom_axes = True):
    '''Plots a spectrogram of signal s onto the given axes.

    Parameters:
        ax: the matplotlib axes to plot onto
        s: the signal (numpy array)
        sampling_rate: samples per second (passed to specgram as Fs)
        title: chart title (optional)
        marker: unused, kept for signature symmetry with plot_signal_to_axes
        custom_axes: if True, relabel the x-axis in samples and add a top time-axis

    Returns:
        the tuple returned by matplotlib's specgram (spectrum, freqs, t, image).
    '''

    specgram_return_data = ax.specgram(s, Fs=sampling_rate)

    # nudge the title up (y=1.2) to make room for the secondary x-axis added below
    if title is not None:
        ax.set_title(title, y=1.2)

    ax.set_ylabel("Frequency")

    # add in a secondary x-axis to draw the x ticks as time (rather than samples)
    if custom_axes:
        ax.set(xlabel="Samples")
        ax_xtick_labels = np.array(ax.get_xticks()) * sampling_rate
        ax_xtick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
        # pin the tick locations before setting labels, else matplotlib warns that the labels
        # may not line up with the ticks (FixedFormatter without a FixedLocator)
        ax.set_xticks(ax.get_xticks())
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
    '''Plots a spectrogram of s (full view) next to a zoomed-in view.

    Parameters:
        s: the signal
        sampling_rate: samples per second
        title: chart title (a duration/rate default is used if None)
        xlim_zoom: optional (start_sample, end_sample) range; a random 10% window is used if None
        highlight_zoom_area: if True, mark the zoomed range on the full-view panel

    Returns:
        (fig, axes, specgram_return_data_full, specgram_return_data_zoom).
    '''
    fig, axes = plt.subplots(1, 2, figsize=(15,4), gridspec_kw={'width_ratios': [2, 1]})

    if title is None:
        # duration in secs = number of samples / samples-per-second
        length_in_secs = len(s) / sampling_rate
        title = f"{length_in_secs:.2f} sec Signal at {sampling_rate} Hz"

    specgram_return_data0 = plot_spectrogram_to_axes(axes[0], s, sampling_rate, title)

    if xlim_zoom is None:
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
    # pin tick locations before relabeling so matplotlib doesn't warn about mismatched ticks/labels
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_xticklabels(ax2_tick_labels_strs)

    if highlight_zoom_area:
        # yellow highlight color: color='#FFFBCC'
        axes[0].axvline(x = zoom_x1, linewidth=2, color='r', alpha=0.8, linestyle='-.')
        axes[0].axvline(x = zoom_x2, linewidth=2, color='r', alpha=0.8, linestyle='-.')
    
    fig.tight_layout()
    return (fig, axes, specgram_return_data0, specgram_return_data1)

def plot_signal_and_spectrogram(s, sampling_rate, quantization_bits, xlim_zoom = None, highlight_zoom_area = True):
    '''Plots the waveform (top) and spectrogram (bottom) together, each with a zoomed-in panel.

    Parameters:
        s: the signal
        sampling_rate: samples per second
        quantization_bits: bit depth, used only to build the title
        xlim_zoom: (start_sample, end_sample) range to show zoomed -- required (the right-hand
            panels zoom to this range)
        highlight_zoom_area: if True, mark the zoomed range on the full-view panels
    '''
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