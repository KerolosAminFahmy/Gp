import os
os.environ['TF-CPP-MIN-LOG-LEVEL']='2'

import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# tf.config.set_visible_devices([],'GPU')
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices())
VIDEO_DURATION_SEC = 2
NUM_FRAMES = 30
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
    print(f"total Frames :{total_frames} ")
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frame = cv2.resize(frame, RESIZE_DIMS)
            # frame=extract_features(frame)
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
    data = np.array(data)
    labels = np.array(labels)
    np.save(f"data.npy", data)
    np.save(f"labels.npy", labels)
    print(data.shape)


    return data, labels

# print(process_video_data("new_data_frames_28 _video"))

# data = np.load("data.npy")
# labels = np.load("labels.npy")
# print(data.shape)
# print(labels.shape)


# data = np.concatenate((data0, data1,data2,data3))
# labels = np.concatenate((labels0, labels1,labels2,labels3))




data0 = np.load("Old Data With Feature Extraction using vggg/0_data.npy")
labels0 = np.load("Old Data With Feature Extraction using vggg/0_labels.npy")
data1 = np.load("Old Data With Feature Extraction using vggg/1_data.npy")
labels1 = np.load("Old Data With Feature Extraction using vggg/1_labels.npy")
data2 = np.load("Old Data With Feature Extraction using vggg/2_data.npy")
labels2 = np.load("Old Data With Feature Extraction using vggg/2_labels.npy")
data3 = np.load("Old Data With Feature Extraction using vggg/3_data.npy")
labels3 = np.load("Old Data With Feature Extraction using vggg/3_labels.npy")
# # data5 = np.load("5_data.npy")
# # labels5 = np.load("5_labels.npy")
# # data6 = np.load("6_data.npy")
# # labels6 = np.load("6_labels.npy")
# # data7 = np.load("7_data.npy")
# # labels7 = np.load("7_labels.npy")
# # data8 = np.load("8_data.npy")
# # labels8 = np.load("8_labels.npy")
# # data9 = np.load("9_data.npy")
# # labels9 = np.load("9_labels.npy")


data=np.concatenate((data0,data1,data2,data3))
labels=np.concatenate((labels0,labels1,labels2,labels3))


def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        # layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        # # layers.BatchNormalization(),
        # layers.Conv2D(64, 3, activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(6,padding='same'),
        # layers.Conv3D(64, 3, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Conv3D(128, 3, activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling3D(2),
        # layers.Conv3D(256, 3, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Conv2D(256, 3, activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(2),
        # layers.Conv2D(512, 3, padding="same", activation='relu'),
        # layers.BatchNormalization(),
        # layers.Conv2D(512, 3, padding="same", activation='relu'),
        # layers.BatchNormalization(),
        # layers.Conv3D(512, 3, padding="same", activation='relu'),
        # layers.BatchNormalization(),
        # layers.Conv2D(256, 3, padding="same", activation='relu',input_shape=input_shape),
        # layers.Conv2D(256, 3, padding="same", activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(2),
        # layers.Reshape((-1,25088)),
        # layers.RepeatVector(),
        # layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        # layers.Attention(),
        # layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        # layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        # layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dense(64, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dense(32, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.5),
        #best try here
        # layers.Reshape((-1,160*150), input_shape=input_shape),
        # layers.LSTM(256, return_sequences=True),
        # layers.Dropout(0.5),
        # layers.LSTM(512, return_sequences=True),
        # layers.Dropout(0.5),
        # layers.LSTM(512, return_sequences=True),
        # layers.Dropout(0.5),
        # layers.LSTM(256, return_sequences=True),
        # layers.Dropout(0.5),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, activation='softmax')
        # layers.Reshape((*input_shape,1)),

        layers.Reshape((-1, 7, 7 * 512)),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128,return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.GRU(64,return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.GRU(64))),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model1=lip_reading_model((60,7,7,512),4)
# print(model1.summary())
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
batch_size = 10
model1.fit(X_train,y_train, epochs=35,batch_size=2)
del X_train
del y_train
print(model1.evaluate(X_test,y_test,batch_size=42))
# def lip_reading_model_again(input_shape=(60, 150, 160), num_classes=4):
#    inputs = layers.Input(shape=(*input_shape, 1))
#    x = layers.ZeroPadding3D(padding=(1, 2, 2))(inputs)
#    x = layers.Conv3D(
#        32, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu",
#    )(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.SpatialDropout3D(0.2)(x)
#    x = layers.MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)
#    for _ in range(1):
#        x = layers.Conv3D(
#                32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"
#            )(x)
#        x =layers.BatchNormalization()(x)
#        x = layers.SpatialDropout3D(0.2)(x)
#        x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)
#        # x = tf.concat([x, y], axis=-1)
#    x=layers.Reshape((-1,19,18*32))(x)
#    x = layers.TimeDistributed(layers.Bidirectional(layers.GRU(32, return_sequences=True)))(x)
#    x = layers.TimeDistributed(layers.Bidirectional(layers.GRU(32)))(x)
#    outputs = layers.TimeDistributed(
#        layers.Dense(num_classes, activation="softmax")
#    )(x)
#
#    model = tf.keras.Model(inputs=inputs, outputs=outputs)
#    model.compile(
#        loss='sparse_categorical_crossentropy',
#        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#        metrics=['accuracy']
#    )
#
#    return model

