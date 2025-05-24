# Import custom configuration and utility functions for EMG signal processing and plotting
from gapwatch_extraction.config import *
from gapwatch_extraction.utils_extraction import *
from gapwatch_extraction.utils_general import *

########################################################################
# Set ROS topic and available bagfile paths
########################################################################
selected_topic = '/emg'  # ROS topic to read EMG messages from

# List of available bag files for EMG recordings
pinch_serra =       'dataset/pinch_serra.bag'
ulnar_serra =       'dataset/ulnar_serra.bag'
power_serra =       'dataset/power_serra.bag'
molto_power_papi =  'dataset/molto_power_papi.bag'
super_power_matti = 'dataset/super_power_matti.bag'
fuck_matti =        'dataset/fuck_matti.bag'
new = 'dataset/2025-05-23-10-32-21.bag'

########################################################################
# Read EMG data from selected ROS bag file
########################################################################

# Initialize list to store EMG data
emg_data = []

# Choose which bag file to load
bag_path = new  # <-- Change here to use a different file

# Open the bag and extract EMG values from messages
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        try:
            for i in msg.emg:  # Read each value in the EMG array
                emg_data.append(i)
        except AttributeError as e:
            print("Message missing expected fields:", e)
            break

# Print the total number of EMG values collected
print(len(emg_data))

########################################################################
# Reshape raw EMG vector into (16 x N) matrix format
########################################################################

# The bag file streams data as a flat list; this section reformats it into 16 channels
selector = 0
final_emg = np.empty((16, 0))  # Initialize empty matrix with 16 rows (channels)

# Loop over all complete sets of 16-channel samples
for i in range(int(len(emg_data)/16)):
    temp = emg_data[selector:selector+16]            # Extract 16 consecutive samples
    new_column = np.array(temp).reshape(16, 1)       # Convert to column format
    final_emg = np.hstack((final_emg, new_column))   # Append column to EMG matrix
    selector += 16                                   # Move to next block
    print("Sample number: ", i)

########################################################################
# Plot raw EMG data
########################################################################

# Show EMG channels in both full and separate subplot formats
plot_emg_channels_2cols(final_emg)   # Plots 16 individual subplots (2 columns)
plot_all_channels(final_emg)         # Plots all 16 channels on the same plot

########################################################################
# PCA Synergy Extraction
########################################################################

"""
# Apply Principal Component Analysis (PCA) to extract synergies from EMG
optimal_synergies_pca = 3
final_emg_for_pca = final_emg.T  # Transpose for sklearn compatibility (samples as rows)

# Decompose EMG into synergy components and reconstruct signal
S_m, U, mean, rec = pca_emg(final_emg_for_pca, optimal_synergies_pca, random_state=42, svd_solver='full')
reconstructed_pca = pca_emg_reconstruction(U, S_m, mean, optimal_synergies_pca)

# Plot original, reconstructed, and synergy data
plot_all_results(final_emg_for_pca, reconstructed_pca, U, S_m, optimal_synergies_pca)
"""

########################################################################
# Sparse NMF Synergy Extraction
########################################################################


# Apply Sparse Non-negative Matrix Factorization (NMF) to extract synergies
optimal_synergies_nmf = 3
final_emg_for_nmf = final_emg.T  # Transpose for sklearn compatibility

# Decompose EMG using sparse NMF into synergies and activation patterns
U, S_m = nmf_emg(final_emg_for_nmf, n_components=optimal_synergies_nmf,
                 init='nndsvd', max_iter=200, l1_ratio=0.7, alpha_W=0.01, random_state=42)

# Reconstruct the EMG from extracted synergies
reconstructed_nmf = nmf_emg_reconstruction(U, S_m, optimal_synergies_nmf)

# Plot original, reconstructed, and synergy data
plot_all_results(final_emg_for_nmf, reconstructed_nmf, U, S_m, optimal_synergies_nmf)



