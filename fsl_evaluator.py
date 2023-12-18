from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback


import numpy as np
import pandas as pd
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

X_train = np.array(sequences)
y_train = to_categorical(labels).astype(int)

model = load_model('model.35v2.h5')

yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

conf_matrix = multilabel_confusion_matrix(ytrue, yhat)

data = []
for label, index in label_map.items():
    row_data = {'Label': label}
    
    # Extract TN, FP, FN, TP values
    tn, fp, fn, tp = conf_matrix[index].ravel()
    
    row_data['TN'] = tn
    row_data['TP'] = tp
    row_data['FP'] = fp
    row_data['FN'] = fn

    
    data.append(row_data)

# Create DataFrame
df_conf_matrix = pd.DataFrame(data)

# Display the DataFrame
print(df_conf_matrix)

# df_conf_matrix.to_csv('confusion_matrix_data_25.csv', index=False)

print(accuracy_score(ytrue, yhat))
