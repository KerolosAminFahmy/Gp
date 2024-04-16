import os

from scipy.odr import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Flatten, Dense, Reshape

os.environ['TF-CPP-MIN-LOG-LEVEL']='2'

import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
VIDEO_DURATION_SEC = 2
NUM_FRAMES = 60
RESIZE_DIMS = (150, 160)
def extract_features(frame):
    model = VGG19(weights='imagenet', include_top=False)
    img = cv2.resize(frame, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=np.int32)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to grayscale
            frame = cv2.resize(frame, RESIZE_DIMS)
            frame=extract_features(frame)
            frames.append(frame)

    cap.release()
    return np.array(frames)
def process_video_data(root_folder):
    data = []
    labels = []
    for label, folder_name in enumerate(sorted(os.listdir(root_folder))):
        folder_path = os.path.join(root_folder, folder_name)
        counter=0
        if os.path.isdir(folder_path):
            print(f"start saving folder {folder_name}")
            for video_file in os.listdir(folder_path):
                counter+=1
                video_path = os.path.join(folder_path, video_file)
                frames = extract_frames(video_path)
                if len(frames) == NUM_FRAMES:
                    print(f"video number saved {counter}")
                    data.append(frames)
                    labels.append(label)
                    np.save(f"{folder_name}_data.npy", data)
                    np.save(f"{folder_name}_labels.npy", labels)
                    return;
        data = np.array(data)
        labels = np.array(labels)
        np.save(f"{folder_name}_data.npy", data)
        np.save(f"{folder_name}_labels.npy", labels)
        print(data.shape)
        print(f"Data for folder '{folder_name}' saved.")
    return data, labels
#process_video_data("new_data_frames_28 _video")


# data=np.load("Old_Data_With_Feature_Extraction_using_vggg/0_data.npy")
# print(data.shape)

# X = np.load("X_train.npy", mmap_mode='r')
# y = np.load("y_train.npy")
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# np.save("X_train_1.npy",X_train)
# np.save("X_val.npy",X_val)
# np.save("Y_train_1.npy",y_train)
# np.save("y_val.npy",y_val)
#
# print(np.load("Old_Data_Without_Feature_Extraction/data.npy").shape)
# print(np.load("Old_Data_Without_Feature_Extraction/labels.npy").shape)
#
# data=np.load("Old_Data_Without_Feature_Extraction/data.npy")
# label=np.load("Old_Data_Without_Feature_Extraction/labels.npy")
print(np.load("Old_Data_Without_Feature_Extraction/X_train.npy").shape)
# X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.33, random_state=42)
# del data,label
# np.save("Old_Data_Without_Feature_Extraction/X_train.npy",X_train)
# del X_train
# np.save("Old_Data_Without_Feature_Extraction/X_test.npy",X_val)
# del X_val
# np.save("Old_Data_Without_Feature_Extraction/Y_train.npy",y_train)
# np.save("Old_Data_Without_Feature_Extraction/y_test.npy",y_val)