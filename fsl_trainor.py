from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


import numpy as np
import os

import os

np.random.seed(42)

data_folder = "data"

# Function to filter folders based on numeric names
def is_numeric_folder(folder_name):
    return folder_name.isdigit()

labels = []
sequences = []
label_map = {
    "ako": 0,
    "maganda": 1,
    "kumain": 2
}

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

# DATA_PATH = 'MP_Data'
# # subfolder_names = ['a', 'b', 'c']
# # print(subfolder_names)
# actions = np.array([
#     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
#     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#     'u','v','w','x','y','z','0','1','2','3','4','5',
#     '6','7','8','9','10'
# ])

# no_sequences = 30
# sequence_length = 30

# label_map = {label:num for num, label in enumerate(actions)}
# sequences, labels = [], []

# print(label_map)

# for action in actions:
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])


# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_map), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X, y, epochs=100, callbacks=[tb_callback])

model.summary()

model.save('action.h5')


