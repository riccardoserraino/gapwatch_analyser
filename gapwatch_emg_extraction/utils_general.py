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
# Functions to filter the data, has been developed 2 approaches: butterworth bandpass and notch

# Band-pass 10-500Hz, Notch 50Hz
# 1. Bandpass filter design
def butter_bandpass(signal, fs, lowcut=20, highcut=500, order=5):
    """Applies a Butterworth bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_b = filtfilt(b, a, signal)
    return filtered_b

# 2. Notch filter design
def notch_filter(signal, fs, n_freq=50.0, Q=30.0):
    """Applies a notch filter to remove powerline interference."""
    nyq = 0.5 * fs
    w0 = n_freq / nyq  
    b, a = iirnotch(w0, Q)
    filtered_n = filtfilt(b, a, signal)
    return filtered_n

# 3. RMS in 200 ms Windows 
def compute_rms(signal, window_size=200):
    """Computes the RMS of the signal using a moving window."""
    # RMS over sliding windows
    squared = np.power(signal, 2)
    window = np.ones(window_size)/window_size
    rms = np.sqrt(np.convolve(squared, window, mode='same'))
    return rms

# 4. Apply bandpass and notch filters to the signal + rms
def preprocess_emg(emg_signal, fs):
    bandpassed = butter_bandpass(emg_signal, fs)
    notch_removed = notch_filter(bandpassed, fs)
    rms_signal = compute_rms(notch_removed)
    return rms_signal


#-------------------------------------------------------------------------------------------
# Function to compute the Moore–Penrose pseudo-inverse of a matrix
def compute_pseudo_inverse(matrix):
    """
    Computes the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        matrix (ndarray): Input matrix of shape (n_samples, n_synergies) or similar.

    Returns:
        pseudo_inverse (ndarray): Pseudo-inverse of the input matrix.
    """
    print("\nComputing pseudo-inverse of the neural matrix W from specimen dataset...")
    pseudo_inverse = np.linalg.pinv(matrix)
    print("Input matrix shape:", matrix.shape)
    print("Pseudo-inverse shape:", pseudo_inverse.shape)
    print("Pseudo-inverse computation completed.\n")
    return pseudo_inverse


