import os
os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

data = np.load("data.npy")
print(data.shape)
for da in data:
    print(da)
labels = np.load("labels.npy")
# data1 = np.load("1_data.npy")
# labels1 = np.load("1_labels.npy")
# data2 = np.load("2_data.npy")
# labels2 = np.load("2_labels.npy")
# data3 = np.load("3_data.npy")
# labels3 = np.load("3_labels.npy")
# data=np.concatenate((data0,data1,data2,data3))
# labels=np.concatenate((labels0,labels1,labels2,labels3))
# del data0
# del data1
# del data2
# del data3
# del labels0
# del labels1
# del labels2
# del labels3
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((*input_shape, 1)),
        layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        layers.ZeroPadding3D((2, 2, 1)),
        layers.Conv3D(128, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(0.2),
        layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        layers.Conv3D(
            1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(0.2),
        layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        layers.Conv3D(
            512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(0.2),
        layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same"), layers.Conv3D(
            128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(0.2),
        layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        layers.Reshape((-1, 4, 3 * 128)),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.GRU(128, return_sequences=True))),
        layers.TimeDistributed(layers.Bidirectional(layers.GRU(128))),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model