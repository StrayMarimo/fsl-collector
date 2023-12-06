from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback


import numpy as np
import os

import os

np.random.seed(42)

data_folder = "MP_Data"
basic_folder= "MP_Data/basic"

# Function to filter folders based on numeric names
def is_numeric_folder(folder_name):
    return folder_name.isdigit()

labels = []
sequences = []

subfolders = [f.name for f in os.scandir(basic_folder) if f.is_dir()]
label_map = {subfolder: index for index, subfolder in enumerate(subfolders)}


print(label_map)

# Iterate through the main data folder
for pos_folder in os.listdir(data_folder):
    pos_folder_path = os.path.join(data_folder, pos_folder)
    # Check if it's a directory
    if os.path.isdir(pos_folder_path):
        # Iterate through subfolders of the current part of speech folder
        for class_folder in os.listdir(pos_folder_path):
            class_folder_path = os.path.join(pos_folder_path, class_folder)
            # print(f"Processing class folder: {class_folder} in {pos_folder}")
            
            for sequence in np.array(os.listdir(os.path.join(pos_folder_path, class_folder))).astype(int):
                window = []
                for frame_num in range(30):
                    res = np.load(os.path.join(pos_folder_path, class_folder, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[class_folder])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(X.shape)
print(y.shape)

print(labels)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'categorical_accuracy' in logs and logs['categorical_accuracy'] > 0.95:
            print('\nReached categorical accuracy of 90%.\nStopping training...\n')
            self.model.stop_training = True


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_map), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X, y, epochs=500, callbacks=[tb_callback, MyCallback()])

model.summary()

model.save('model.h5')


