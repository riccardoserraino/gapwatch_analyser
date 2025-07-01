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
s_power = "dataset/power_grasp1.bag"
s_pinch = "dataset/pinch1.bag"
s_ulnar = "dataset/ulnar1.bag"
s_thumb_up = "dataset/thumb_up1.bag"
s_sto = "dataset/sto1.bag"

s_bottle = "dataset/bottle1.bag"
s_pen = "dataset/pen1.bag"
s_phone = "dataset/phone1.bag"
s_tablet = "dataset/tablet1.bag"
s_pinza = "dataset/pinza1.bag"

s_thumb = "dataset/thumb1.bag"
s_index = "dataset/index1.bag"
s_middle = "dataset/middle1.bag"
s_ring = "dataset/ring1.bag"
s_little = "dataset/little1.bag"

#-----------------------------------------------------------------------
# Test bag files (optional)
power = "dataset/power_grasp2.bag"
pinch = "dataset/pinch2.bag"
ulnar = "dataset/ulnar2.bag"
thumb_up = "dataset/thumb_up2.bag"
sto = "dataset/sto2.bag"

bottle = "dataset/bottle2.bag"
pen = "dataset/pen2.bag"
phone = "dataset/phone2.bag"
tablet = "dataset/tablet2.bag"
pinza = "dataset/pinza2.bag"

thumb = "dataset/thumb2.bag"
index = "dataset/index2.bag"
middle = "dataset/middle2.bag"
ring = "dataset/ring2.bag"
little = "dataset/little2.bag"


########################################################################
# Data loading - Read EMG data from selected ROS bag file
########################################################################

#-----------------------------------------------------------------------
# Load EMG data from a specific bag file
# Initialize list to store EMG data
emg_data_specimen = []
timestamps_specimen = []

# Choose which bag file to load for specimen and analysis
bag_path_specimen = s_ring     # <-- Change here to use a different file
#bag_path_test = power

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

# Convert to numpy arrays
emg_data_specimen = np.array(emg_data_specimen)
timestamps_specimen = np.array(timestamps_specimen)

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

# Print shape information of extracted data
print(f"Acquired EMG data shape: {emg_data_specimen.shape}")  # Should be (n_samples, n_channels)

reshaped_timestamps = timestamps_specimen[::16]
reshaped_timestamps_int = len(reshaped_timestamps)
print(f"Reshaped timestamps shape: {reshaped_timestamps.shape}")  
print(f"Timestamps count: {reshaped_timestamps_int}")
duration = reshaped_timestamps[-1] - reshaped_timestamps[0]
print(f"Duration of EMG recording: {duration:.5f} s")

# Print shape information of reshaped data
print("Final EMG shape:", raw_emg.shape)


#-----------------------------------------------------------------------
# Test reshaping (optional)



########################################################################
# Data filtering - Filtering the raw data to remove noise and baseline drift
########################################################################

#-----------------------------------------------------------------------
# Specimen filtering
s_fs=reshaped_timestamps_int/duration
print(f"Sampling frequency fs = {s_fs:.2f} Hz")

# Band-pass + Notch filtering + rms
filtered_emg= np.array([preprocess_emg(raw_emg[i, :], fs=s_fs) for i in range(raw_emg.shape[0])])


#-----------------------------------------------------------------------
# Test filtering (optional)


########################################################################
# Data Plotting - Plot first insights into EMG data aquired from ROS bag (optional)
########################################################################

# Plot all raw channels in a single plot
#plot_all_channels(raw_emg, title='Raw EMG Channels')         
#plot_emg_channels_2cols(raw_emg)

# Filtering insights section--------------------------------------------
# Plotting with filter applied to raw data
#plot_raw_vs_filtered_channels_2cols(raw_emg, filtered_emg, title='Raw vs Band-pass & Notch Filtered EMG Channels')
#plot_all_channels(filtered_emg, title='Band-pass & Notch Filtered EMG Channels')
plot_emg_channels_2cols(filtered_emg)



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
                 init='nndsvd', max_iter=500, l1_ratio=0.15, alpha_W=0.0005, random_state=42)

# Reconstruct the EMG from extracted synergies
reconstructed_nmf = nmf_emg_reconstruction(W, H, optimal_synergies_nmf)

# Plot original, reconstructed, and synergy data
plot_all_results(final_emg_for_nmf, reconstructed_nmf, W, H, optimal_synergies_nmf, title='NMF Synergy Extraction Results - Tablet Grasp')




########################################################################

# Pseudo inverse of H matrix (neural matrix representing activation patterns)
W_pinv = compute_pseudo_inverse(W)

estimated_H = np.dot(W_pinv, final_emg_for_nmf)
print("Estimated Synergy Matrix H from W_pinv shape:", estimated_H.shape)   




