from gapwatch_emg_extraction.config import *
from gapwatch_emg_extraction.utils_general import *

###########################################################################################################
# Visualize helper functions
###########################################################################################################


#-------------------------------------------------------------------------------------------
def plot_all_channels(emg_data, title=''):
    """
    Plot all EMG channels on a single graph for a global overview.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_channels, n_samples).

    Returns:
        None. Displays a matplotlib figure.
    """

    plt.figure(figsize=(8, 6)) 
    for j in range(emg_data.shape[0]):
        x = np.linspace(0, emg_data.shape[1] , emg_data.shape[1])
        plt.plot(x, emg_data[j], label='Channel {}'.format(j))
    plt.title("EMG signal overview")
    plt.xlabel("Samples over time")
    plt.ylabel("Channel activation")
    if title: 
        plt.title(title)
    plt.legend(loc='best', fontsize='small', markerscale=1)
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_emg_channels_2cols(emg_data):
    """
    Plot each EMG channel in a separate subplot, organized into two columns.

    Args:
        emg_data (ndarray): 2D array of EMG data with shape (n_channels, n_samples).

    Returns:
        None. Displays a matplotlib figure with subplots for each channel.
    """

    n_channels, n_samples = emg_data.shape

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(8, 8), sharex=True)
    time = np.linspace(0, n_samples, n_samples)

    for i in range(16):
        row = i % 8
        col = i // 8
        ax = axes[row, col]
        ax.plot(time, emg_data[i])
        ax.set_title(f'Channel {i}')
    
        if row == 7:
            ax.set_ylabel("Activation")
            ax.set_xlabel("Time (samples)")
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg):
    """
    Plot raw and filtered EMG signals in subplots, organized into two columns.

    Args:
        raw_emg (ndarray): Raw EMG data, shape (n_channels, n_samples)
        filtered_emg (ndarray): Filtered EMG data, same shape

    Returns:
        None. Displays matplotlib figure.
    """

    n_channels, n_samples = raw_emg.shape
    time = np.linspace(0, n_samples, n_samples)

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(10, 10), sharex=True)

    for i in range(n_channels):
        row = i % 8
        col = i // 8
        ax = axes[row, col]

        ax.plot(time, raw_emg[i], label='Raw', alpha=0.6)
        ax.plot(time, filtered_emg[i], label='Filtered', linestyle='--')
        ax.set_title(f'Channel {i}', fontsize=8)

        if row == 7:
            ax.set_xlabel("Time (samples)")
        if col == 0:
            ax.set_ylabel("Activation")

        ax.legend(fontsize=6)
    fig.suptitle("Filtered vs Raw - EMG Channels Overview")

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
def plot_all_results(emg_data, Z_reconstructed, U, S_m, selected_synergies):
    """
    Plot a comprehensive overview of EMG signal decomposition using synergies.

    This function generates four stacked subplots:
    1. Original EMG signals.
    2. Reconstructed EMG signals from synergies.
    3. Time-varying activation of each synergy.
    4. Synergy-to-muscle weight distributions.

    Args:
        emg_data (ndarray): Raw EMG data (n_samples x n_muscles).
        Z_reconstructed (ndarray): Reconstructed EMG from synergy model (same shape).
        U (ndarray): Synergy activation matrix (n_samples x n_synergies).
        S_m (ndarray): Synergy weights matrix (n_synergies x n_muscles).
        selected_synergies (int): Number of synergies used in the model.

    Returns:
        None. Displays a matplotlib figure with 4 subplots.
    """
    
    print(f'\nPlotting results...\n\n')

    U_scaled = scale_synergy_signal(U, emg_data)


    plt.figure(figsize=(10, 8))
    
    # Panel 1: Original EMG Signals
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.title('Original EMG Signals')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization
    
    # Panel 2: Reconstructed EMG Signals
    plt.subplot(4, 1, 2)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.title(f'Reconstructed EMG ({selected_synergies} Synergies)')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])
    
    # Panel 3: Synergy Activation Patterns over time
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(U_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    # Panel 4: Synergy Weighting Patterns
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(S_m[i, :], 'o-', label=f'Synergy {i+1}')
    plt.title('Synergy Weighting Patterns')
    plt.xlabel('EMG Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])

    plt.tight_layout()
    plt.show()


