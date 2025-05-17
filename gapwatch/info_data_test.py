import rosbag
import numpy as np
import matplotlib.pyplot as plt


bag_path = 'dataset/try.bag'
selected_topic = '/emg'


with rosbag.Bag(bag_path, 'r') as bag:
    info = bag.get_type_and_topic_info()


# Print available topics and their message types
print("==========================")
print("Bag file information:")
print("\nAvailable topics and message types:\n")

for topic, details in info.topics.items():
    print(f"- Topic: {topic}")
    print(f"  Type: {details.msg_type}")
    print(f"  Messages: {details.message_count}")
print("\n==========================")



timestamps = []
emg_data = []
battery = []
counter = []

with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[selected_topic]):
        timestamps.append(t.to_sec())

        try:
            emg_data.append(list(msg.emg))
            battery.append(msg.battery)
            counter.append(msg.counter)
        except AttributeError as e:
            print("Message missing expected fields:", e)
            break


timestamps = np.array(timestamps)
emg_data = np.array(emg_data)
battery = np.array(battery)
counter = np.array(counter)



# === Print shapes ===
print("\nShape of data:\n")
print("Timestamps:", timestamps.shape)
print("EMG data:", emg_data.shape)
print("Battery:", battery.shape)
print("Counter:", counter.shape)
print("\n===========================")


print("\nPlotting Data\n")
# === Plot EMG signal (first 2 channels) ===
plt.figure(figsize=(10, 5))
plt.plot(emg_data )
plt.xlabel('Time (s)')
plt.ylabel('EMG Value')
plt.title('EMG Signal Overview')
plt.tight_layout()
plt.show()

# === Optional: Plot battery and counter ===
plt.figure(figsize=(10, 3))
plt.plot(battery, label='Battery Level', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Battery')
plt.title('Battery Level Over Time')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(counter, label='Counter', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Counter')
plt.title('Counter Over Time')
plt.tight_layout()
plt.show()



print("============================")
print("\nEnd of Session\n")