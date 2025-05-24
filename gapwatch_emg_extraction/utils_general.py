from gapwatch_emg_extraction.config import *

###########################################################################################################
# General purpose functions
###########################################################################################################


#-------------------------------------------------------------------------------------------
# Function to scale the synergy activation matrix to the original EMG amplitude range

def scale_synergy_signal(X, emg_data):
    """
    Normalize synergy activation matrix to the amplitude range of the original EMG.

    This ensures that synergy activations (X) can be compared or plotted in the 
    same scale as EMG signals.

    Args:
        X (ndarray): Activation matrix (n_samples x n_synergies).
        emg_data (ndarray): Original EMG signals (n_samples x n_channels).

    Returns:
        ndarray: Scaled activation matrix (same shape as X).
    """
    
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = ((X - X_min) / (X_max - X_min)) * (emg_max - emg_min) + emg_min
    X_scaled = np.maximum(X_scaled, 0)  # Ensures W_scaled is non-negative
    return X_scaled


#-------------------------------------------------------------------------------------------
# Function to filter the data

def low_pass_filter(signal, cutoff=10, fs=1000, order=4):
    """
    Apply a low-pass Butterworth filter to 1D signal.
    
    Parameters:
    - signal: array-like
    - cutoff: cutoff frequency (Hz)
    - fs: sampling rate (Hz)
    - order: filter order

    Returns:
    - Filtered signal
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


#-------------------------------------------------------------------------------------------
# Function to align baselines of multiple signals

def align_signal_baselines(signals, method='mean'):
    """
    Aligns all signals to the same baseline.

    Parameters:
    - signals: list of 1D numpy arrays
    - method: 'mean', 'first', or 'min'

    Returns:
    - list of aligned signals
    """
    if method == 'mean':
        offsets = [np.mean(s) for s in signals]
    elif method == 'first':
        offsets = [s[0] for s in signals]
    elif method == 'min':
        offsets = [np.min(s) for s in signals]
    else:
        raise ValueError("Invalid method")

    reference = np.mean(offsets)  # Align to the group mean
    return [s - (off - reference) for s, off in zip(signals, offsets)]


#-------------------------------------------------------------------------------------------

