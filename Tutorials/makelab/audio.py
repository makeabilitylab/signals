"""Audio helpers for the Tutorials notebooks.

Currently just stereo->mono conversion. Audio *loading* (librosa/soundfile) happens
in the notebooks themselves; this module only post-processes the arrays they produce.
"""
import numpy as np  # numpy: https://numpy.org/

def convert_to_mono(audio_data):
    '''Converts stereo audio to mono by averaging across channels.

    Parameters:
        audio_data (np.ndarray): audio samples. A 2-D array is treated as
            (num_samples, num_channels); a 1-D array is assumed already-mono.

    Returns:
        np.ndarray: a 1-D mono signal. Mono input is returned unchanged.

    Note: librosa offers `librosa.to_mono(y)` for the same job, but it expects a
    *channels-first* array of shape (num_channels, num_samples) -- the transpose of
    the channels-last layout the notebooks load -- so we average ourselves here.
    '''
    if len(audio_data.shape) == 2:
        print("Converting stereo audio file to mono")
        # mean(axis=1) averages across channels and works for any channel count
        # (the old `sum(axis=1) / 2` silently assumed exactly two channels).
        return audio_data.mean(axis=1)
    return audio_data
