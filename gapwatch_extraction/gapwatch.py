from gapwatch_extraction.utils import *

########################################################################
# Path and topic 
selected_topic = '/emg'

pinch_serra =       'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/pinch_serra.bag'
ulnar_serra =       'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/ulnar_serra.bag'
power_serra =       'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/power_serra.bag'
molto_power_papi =  'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/molto_power_papi.bag'
super_power_matti = 'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/super_power_matti.bag'
fuck_matti =        'C:/Users/ricca/Desktop/th_unibo/code/gapwatch/dataset/fuck_matti.bag'

########################################################################

# Initialize final datasets lists
# timestamps = []
emg_data = []

# Set chosen bag file
bag_path = super_power_matti

# Reading data from the bag file and setting up data
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        # timestamps.append(t.to_sec())
    
        try:
            for i in msg.emg:
                emg_data.append(i)

        except AttributeError as e:
            print("Message missing expected fields:", e)
            break

# Check emg data length
print(len(emg_data))



# Algorithm to reshape data: from 1 line of concatenated data to 16 channel data

selector = 0
final_emg = np.empty((16,0))

for i in range(int(len(emg_data)/16)):
    temp = emg_data[selector:selector+16]

    new_column = np.array(temp).reshape(16,1)
    final_emg = np.hstack((final_emg, new_column))

    selector += 16
    print("Sample number: ", i)

    #print(final_emg)


#------------------------------------------------------------------------------------------------------------------------------------

# Plotting raw data
plot_emg_channels_2cols(final_emg)
plot_all_channels(final_emg)


# PCA application
optimal_synergies_pca = 3
final_emg_for_pca = final_emg.T # Transpose to have samples as rows and channels as columns, better for pca analysis and plotting
S_m, U, mean, rec = pca_emg(final_emg_for_pca, optimal_synergies_pca, random_state=42, svd_solver='full')
reconstructed = pca_emg_reconstruction(U, S_m, mean, optimal_synergies_pca)
final_emg_np = final_emg if isinstance(final_emg, np.ndarray) else final_emg.numpy() # needed for plotting
plot_all_results(final_emg_for_pca, reconstructed, U, S_m, optimal_synergies_pca)




