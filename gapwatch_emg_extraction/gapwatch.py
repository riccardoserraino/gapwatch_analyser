# Import custom configuration and utility functions for EMG signal processing and plotting
from gapwatch_emg_extraction.config import *
from gapwatch_emg_extraction.utils_extraction import *
from gapwatch_emg_extraction.utils_general import *
from gapwatch_emg_extraction.utils_visual import *



########################################################################
# Initialization - Set ROS topic and available bagfile paths
########################################################################
selected_topic = '/emg'  # ROS topic to read EMG messages from

# List of available bag files for EMG recordings
#-----------------------------------------------------------------------
# Specimen bag files
power_grasp = 'dataset/power_grasp.bag'

pinch_grasp = 'dataset/pinch_grasp.bag'
ulnar_grasp = 'dataset/ulnar_grasp.bag'

test_con_mattia = 'dataset/test_con_mattia.bag'

#-----------------------------------------------------------------------
# Test bag files (optional)



########################################################################
# Data loading - Read EMG data from selected ROS bag file
########################################################################

#-----------------------------------------------------------------------
# Load EMG data from a specific bag file
# Initialize list to store EMG data
emg_data_specimen = []
timestamps_specimen = []

# Choose which bag file to load for specimen and analysis
bag_path_specimen = ulnar_grasp            # <-- Change here to use a different file
#bag_path_test = test_con_mattia

# Open the bag and extract EMG values from messages
with rosbag.Bag(bag_path_specimen, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        try:
            for i in msg.emg:  # Read each value in the EMG array
                emg_data_specimen.append(i)
                timestamps_specimen.append(t.to_sec())
        except AttributeError as e:
            print("Message missing expected fields:", e)
            break

# Print the total number of EMG values collected
print(len(emg_data_specimen))

# Print the recording duration in seconds (needed for the filtering process)
duration = timestamps_specimen[-1] - timestamps_specimen[0]
print(f"Recording duration: {duration} seconds")


#-----------------------------------------------------------------------
# Load EMG data from a test bag file (optional)



########################################################################
# Data Processing - Reshape raw EMG vector into (16 x N) matrix format
########################################################################
# The bag file streams data as a flat list; this section reformats it into 16 channels

#-----------------------------------------------------------------------
# Specimen reshaping
selector = 0
raw_emg = np.empty((16, 0))  # Initialize empty matrix with 16 rows (channels)

# Loop over all complete sets of 16-channel samples
for i in range(int(len(emg_data_specimen)/16)):
    temp = emg_data_specimen[selector:selector+16]            # Extract 16 consecutive samples
    new_column = np.array(temp).reshape(16, 1)       # Convert to column format
    raw_emg = np.hstack((raw_emg, new_column))   # Append column to EMG matrix
    selector += 16                                   # Move to next block
    print("Sample number: ", i)

# Print the shape of the final EMG matrix
print("Final EMG shape:", raw_emg.shape)
# Print the first channel shape
channel_shape = raw_emg[0].shape[0]
print("First channel shape:", channel_shape)  # Should be (N,)

#-----------------------------------------------------------------------
# Test reshaping (optional)



########################################################################
# Data filtering - Filtering the raw data to remove noise and baseline drift
########################################################################

#-----------------------------------------------------------------------
# Specimen filtering
fs=channel_shape/duration
print("Sampling frequency fs =", fs)

# Band-pass + Notch filtering + rms
filtered_emg= np.array([preprocess_emg(raw_emg[i, :], fs=fs) for i in range(raw_emg.shape[0])])


#-----------------------------------------------------------------------
# Test filtering (optional)
'''
# Butterworth filtering
butt_filtered_emg_ch = np.array([butterworth_filter(raw_emg[i, :], cutoff=10, fs=fs) for i in range(raw_emg.shape[0])])
butt_filtered_aligned_emg = np.array(align_signal_baselines(butt_filtered_emg_ch, method='mean'))
'''


########################################################################
# Data Plotting - Plot first insights into EMG data aquired from ROS bag (optional)
########################################################################
'''
# Butterworth insights section------------------------------------------
# Plot raw vs butterworth filtered EMG data
plot_raw_vs_filtered_channels_2cols(raw_emg, butt_filtered_emg_ch, title='Raw vs Lowpass Butterworth Filtered EMG Channels')
plot_all_channels(butt_filtered_aligned_emg, title='Lowpass Butterworth Filtered EMG Channels')
'''


# Plot all raw channels in a single plot
#plot_all_channels(raw_emg, title='Raw EMG Channels')         
#plot_emg_channels_2cols(raw_emg)

# Filtering insights section--------------------------------------------
# Plotting with filter applied to raw data
#plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg, title='Raw vs Band-pass & Notch Filtered EMG Channels')
#plot_all_channels(filtered_emg, title='Band-pass & Notch Filtered EMG Channels')
#plot_emg_channels_2cols(filtered_emg)





########################################################################
# PCA Synergy Extraction (optional)
########################################################################

'''
# Apply Principal Component Analysis (PCA) to extract synergies from EMG (filtered data, no alignment introduced)
optimal_synergies_pca = 2
max_components_pca = 16
final_emg_for_pca = filtered_emg.T  # Transpose for sklearn compatibility (samples as rows)

# Decompose EMG into synergy components and reconstruct signal
H, W, mean, rec = pca_emg(final_emg_for_pca, optimal_synergies_pca, random_state=42, svd_solver='full')
reconstructed_pca = pca_emg_reconstruction(W, H, mean, optimal_synergies_pca)

# Plot original, reconstructed, and synergy data
plot_all_results(final_emg_for_pca, reconstructed_pca, W, H, optimal_synergies_pca)
'''



########################################################################
# Sparse NMF Synergy Extraction
########################################################################


# Apply Sparse Non-negative Matrix Factorization (NMF) to extract synergies from EMG (filtered data, no alignment introduced)
optimal_synergies_nmf = 2
max_synergies_nmf = 16
final_emg_for_nmf = filtered_emg.T  # Transpose for sklearn compatibility

# Decompose EMG using sparse NMF into synergies and activation patterns
W, H = nmf_emg(final_emg_for_nmf, n_components=optimal_synergies_nmf,
                 init='nndsvd', max_iter=500, l1_ratio=0.1, alpha_W=0.001, random_state=42)

# Reconstruct the EMG from extracted synergies
reconstructed_nmf = nmf_emg_reconstruction(W, H, optimal_synergies_nmf)

# Plot original, reconstructed, and synergy data
#plot_all_results(final_emg_for_nmf, reconstructed_nmf, W, H, optimal_synergies_nmf, title='NMF Synergy Extraction Results')




########################################################################
'''
# Pseudo inverse of U matrix (neural matrix H representing activation patterns)
U_pinv = compute_pseudo_inverse(U)

estimated_S_m = np.dot(U_pinv, final_emg_for_nmf)
print("Estimated Synergy Matrix S_m from U_pinv shape:", estimated_S_m.shape)   
'''



