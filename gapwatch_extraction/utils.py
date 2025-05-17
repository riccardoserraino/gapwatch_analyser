###########################################################################################################
# Libraries
###########################################################################################################


# General purpose 
import rosbag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NMF implementation
from sklearn.decomposition import NMF

# Autoencoder implementation
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# PCA implementation
from sklearn.decomposition import PCA







###########################################################################################################
# General purpose functions
###########################################################################################################

# Plot all channels in a single plot
def plot_all_channels(emg_data):
    plt.figure(figsize=(8, 6)) 
    for j in range(emg_data.shape[0]):
        x = np.linspace(0, emg_data.shape[1] , emg_data.shape[1])
        plt.plot(x, emg_data[j], label='Channel {}'.format(j))
    plt.title("EMG signal overview")
    plt.xlabel("Samples over time")
    plt.ylabel("Channel activation")
    plt.legend(loc='best', fontsize='small', markerscale=1)
    plt.show()


# Plot each channel separately in two columns
def plot_emg_channels_2cols(emg_data):
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


# Function to filter the data
def averaging(array, point_to_avarage):
    i_prev = 0
    i = 0
    while i < array.shape[0]:
        i = i + point_to_avarage
        mean = np.mean(array[i_prev : i])
        array[i_prev : i] = mean
        i_prev = i
    return array

def scale_synergy_signal(X, emg_data):
    """
    Normalizes synergy activations to match original EMG amplitude range.
    
    Args:
        X: Activation matrix (n_samples x n_sEMG)
        emg_data: Original EMG (n_samples x n_sEMG)
    """
    
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = ((X - X_min) / (X_max - X_min)) * (emg_max - emg_min) + emg_min
    X_scaled = np.maximum(X_scaled, 0)  # Ensures W_scaled is non-negative
    return X_scaled


def plot_all_results(emg_data, Z_reconstructed, U, S_m, selected_synergies):
    """
    Visualizes EMG analysis results in a 4-panel comparative plot.
    
    Args:
        emg_data: Raw EMG (n_samples x n_muscles)
        Z_reconstructed: Reconstructed signal (n_samples x n_muscles)
        H_scaled: Normalized activations (n_samples x n_synergies) (U or H)
        W: Synergy components (n_synergies x n_muscles) (S_m or W)
        selected_synergies: Number of synergies 
        
    Produces:
        Interactive matplotlib figure with:
        1. Original EMG signals
        2. Reconstructed signals
        3. Synergy activations over time (U or H)
        4. Muscle weightings per synergy (S_m or W)
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


###########################################################################################################
# Data processing purpose functions
###########################################################################################################

# PCA functions

def pca_emg(emg_data, n_components, scale_U=False, random_state=None, svd_solver='full'):
    """
    Applies Principal Component Analysis (PCA).
    
    Args:
        emg_data: Input EMG data (n_samples x n_muscles)
        n_components: Number of synergies to extract
        random_state: Random seed for reproducibility
        scale_scores: Whether to scale scores by explained variance
        svd_solver: intialization of the svd method used ('full is the standard LAPACK solver)
    
    Outputs:
        components (S_m): Principal components (n_components x n_muscles) (Muscle activation weight)
        scores (U): Projection of data onto components (n_samples x n_components) (Synergies over time)
        explained_variance: Variance explained by each component
    """

    print("\nApplying PCA...")

    #preprocessing data for a clear reconstruction (centering and normalization)
    X_centered = emg_data - np.mean(emg_data, axis=0)
    X_normalized = (emg_data - np.mean(emg_data, axis=0)) / np.std(emg_data, axis=0)

    #model pca
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    
    #extracting matrices
    U = pca.fit_transform(emg_data)             # Synergies over time
    S_m = pca.components_                       # Muscle patterns (Muscular synergy matrix)
    mean = pca.mean_
    
    # Scale scores by explained variance (makes them more comparable)
    if scale_U:
        U = U * np.sqrt(pca.explained_variance_ratio_)
    # Transpose to keep same structure as NMF function
    if S_m.shape[0] != n_components:
        S_m = S_m.T     # Ensure S_m has shape (n_synergies, n_muscles)
    
    #reconstruction based on the inverse transform
    X_transformed = pca.fit_transform(X_centered) # Neural matrix (synergies over time) adjusted for centering wrt original data and enforce positive values
    X_reconstructed = pca.inverse_transform(X_transformed) + np.mean(emg_data, axis=0) # the mean is added to enforce values of synergies and reconstruction being non negative as the original data

    """mse = np.mean((emg_data - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse}")"""

    print("PCA completed.\n")

    return S_m, U, mean, X_reconstructed


def pca_emg_reconstruction(U, S_m, mean, n_components):
    """
        Reconstruct data using selected number of components.
        
        Returns:
        - reconstructed: Reconstructed data matrix
    """

    print("\nReconstructing original data...")
    
    # Select the first n_components
    U_rec = U[:, :n_components]
    S_m_rec = S_m[:n_components, :]
    
    # Reconstruct the data
    reconstructed = np.dot(U_rec, S_m_rec) + mean

    return reconstructed

#-------------------------------------------------------------------------------------------


# NMF functions

def nmf_emg(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    """
    Applies Non-negative Matrix Factorization (NMF) to EMG data.

    Args:
        emg_data: Input EMG data (n_samples x n_muscles).
        n_components: Number of synergies to extract.
        init: Initialization method for NMF. 
        max_iter: Maximum number of iterations for NMF.
        l1_ratio: L1 ratio for sparse NMF.
        alpha_W: Regularization parameter for U matrix in NMF.
        random_state: Random seed for reproducibility.

    Outputs:
        U: Synergy activations over time (Neural drive matrix)
        S_m: Muscle patterns (Muscular synergy matrix)

    """
    # Pushing initial negative data to 0 for NMF processing
    emg_data_non_negative = np.maximum(0, emg_data)  # Ensure all values are non-negative
    
    print("\nApplying NMF...")
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    U = nmf.fit_transform(emg_data_non_negative)         # Synergy activations over time (Neural drive matrix)
    S_m = nmf.components_                   # Muscle patterns (Muscular synergy matrix)
    
    # Transpose W and H to match the correct shapes if needed
    if U.shape[0] != emg_data_non_negative.shape[0]:
        U = U.T         # Ensure U has shape (n_samples, n_synergies)
    if S_m.shape[0] != n_components:
        S_m = S_m.T     # Ensure S_m has shape (n_synergies, n_muscles)
    print("NMF completed.\n")
    return U, S_m


def nmf_emg_reconstruction(U, S_m, n_synergies):
    print(f"\nReconstructing the signal with {n_synergies} synergies...")
    # Select the first n_synergies components
    U_rec = U[:, :n_synergies]
    S_m_rec = S_m[:n_synergies, :]

    # Reconstruct the original data from the selected components
    reconstructed = np.dot(U_rec, S_m_rec)
    print("Reconstruction completed.\n")
    return reconstructed


#--------------------------------------------------------------------------------------------








