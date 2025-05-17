import rosbag
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################
# Functions
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

###########################################################################################################


# Initialization
selected_topic = '/emg'

bag_path_try = 'dataset/2025-05-15-08-06-10.bag'

pinch_serra = 'dataset/pinch_serra.bag'
ulnar_serra = 'dataset/ulnar_serra.bag'
power_serra = 'dataset/power_serra.bag'
molto_power_papi = 'dataset/molto_power_papi.bag'
super_power_matti = 'dataset/super_power_matti.bag'
fuck_matti = 'dataset/fuck_matti.bag'


# timestamps = []
emg_data = []


# Set chosen bag file
bag_path = fuck_matti

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


# Algorithm to reshape data: from 1 line of 16*number_of_samples (concatenated data) to 16 channel data

selector = 0
final_emg = np.empty((16,0))

for i in range(int(len(emg_data)/16)):
    temp = emg_data[selector:selector+16]

    new_column = np.array(temp).reshape(16,1)
    final_emg = np.hstack((final_emg, new_column))

    selector += 16
    print("Sample number: ", i)

    #print(final_emg)


# Plotting
plot_emg_channels_2cols(final_emg)
plot_all_channels(final_emg)
