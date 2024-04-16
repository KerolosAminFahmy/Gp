import os
os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import cv2
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
VIDEO_DURATION_SEC = 2
NUM_FRAMES = 60
RESIZE_DIMS = (150, 160)
def extract_features(frame):
    model = VGG19(weights='imagenet', include_top=False)
    img = cv2.resize(frame, (224, 224))
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
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            #frame = cv2.resize(frame, RESIZE_DIMS)
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
                    print(counter)
                    data.append(frames)
        data = np.array(data)

        np.save(f"{folder_name}_data.npy", data)
        print(data.shape)


    return data

process_video_data("Dataset_mouth_extraction")




