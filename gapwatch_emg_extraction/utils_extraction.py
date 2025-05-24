from gapwatch_emg_extraction.config import *


####################################################################################################
# Synergies extraction and reconstruction fucntions
####################################################################################################


####################################################################################################
# PCA functions
####################################################################################################

#---------------------------------------------------------------------------------------------
def pca_emg(emg_data, n_components, scale_U=False, random_state=None, svd_solver='full'):
    """
    Applies Principal Component Analysis (PCA) to EMG data for dimensionality reduction 
    and synergy extraction.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_samples, n_muscles).
        n_components (int): Number of principal components (synergies) to extract.
        scale_U (bool): If True, scale scores (U) by the explained variance ratio.
        random_state (int or None): Random seed for reproducibility.
        svd_solver (str): SVD solver to use. Default is 'full' (LAPACK-based).

    Returns:
        S_m (ndarray): Principal components (muscle synergies), shape (n_components, n_muscles).
        U (ndarray): Projection of data onto components (temporal activations), shape (n_samples, n_components).
        mean (ndarray): Mean of the original data used for reconstruction.
        X_reconstructed (ndarray): Reconstructed EMG data using the selected principal components.
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



#---------------------------------------------------------------------------------------------
def pca_emg_reconstruction(U, S_m, mean, n_components):
    """
    Reconstructs EMG data using a selected number of PCA components.

    Args:
        U (ndarray): Scores matrix (temporal activations), shape (n_samples, total_components).
        S_m (ndarray): Principal components (muscle synergies), shape (total_components, n_muscles).
        mean (ndarray): Mean vector used for centering during PCA.
        n_components (int): Number of components to use for reconstruction.

    Returns:
        reconstructed (ndarray): Reconstructed EMG data matrix, shape (n_samples, n_muscles).
    """

    print("\nReconstructing original data...")
    
    # Select the first n_components
    U_rec = U[:, :n_components]
    S_m_rec = S_m[:n_components, :]
    
    # Reconstruct the data
    reconstructed = np.dot(U_rec, S_m_rec) + mean

    return reconstructed



####################################################################################################
# NMF functions
####################################################################################################

#---------------------------------------------------------------------------------------------
def nmf_emg(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    """
    Applies Non-negative Matrix Factorization (NMF) to extract muscle synergies 
    and their activations from EMG data.

    Args:
        emg_data (ndarray): Input EMG data matrix of shape (n_samples, n_muscles).
        n_components (int): Number of synergies (components) to extract.
        init (str): Initialization method for NMF (e.g., 'nndsvd', 'random').
        max_iter (int): Maximum number of iterations before stopping.
        l1_ratio (float): L1 regularization ratio (between 0 and 1).
        alpha_W (float): Regularization strength for the activation matrix U.
        random_state (int): Random seed for reproducibility.

    Returns:
        U (ndarray): Synergy activations over time (neural drive), shape (n_samples, n_components).
        S_m (ndarray): Muscle synergy matrix (muscle weights), shape (n_components, n_muscles).
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



#---------------------------------------------------------------------------------------------
def nmf_emg_reconstruction(U, S_m, n_synergies):
    """
    Reconstructs the EMG signal using a selected number of NMF components (synergies).

    Args:
        U (ndarray): Neural drive matrix (temporal activations), shape (n_samples, total_synergies).
        S_m (ndarray): Muscle synergy matrix (muscle weights), shape (total_synergies, n_muscles).
        n_synergies (int): Number of synergies to use for reconstruction.

    Returns:
        reconstructed (ndarray): Reconstructed EMG data, shape (n_samples, n_muscles).
    """

    print(f"\nReconstructing the signal with {n_synergies} synergies...")
    # Select the first n_synergies components
    U_rec = U[:, :n_synergies]
    S_m_rec = S_m[:n_synergies, :]

    # Reconstruct the original data from the selected components
    reconstructed = np.dot(U_rec, S_m_rec)
    print("Reconstruction completed.\n")
    return reconstructed










