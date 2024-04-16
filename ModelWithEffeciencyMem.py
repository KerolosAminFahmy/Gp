import os

from tensorflow.python.keras.layers import Average

os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((*input_shape,1)),
        layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        layers.ZeroPadding3D((2,2,1)),
        layers.Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(0.2),
        layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        layers.Reshape((-1, 62, 2 * 32)),
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

# data = np.load("data.npy")
labels = np.load("labels.npy")
# Determine index each class and put into array
num_classes = 41
class_subsets = [np.where(labels == i)[0] for i in range(num_classes)]
batch_size = 10
print(labels)
print(num_classes)
# for i in range(0, num_classes, 5):
#     class_indices = np.concatenate(class_subsets[i:i + 5])
#     subset_labels = labels[:len(class_indices)-1]
#     subset_data = data[:len(class_indices)-1]
#     data = data[len(subset_data)-1:]
#     labels = labels[len(subset_labels)-1:]
#     print(np.array(subset_data).shape)
#     print(subset_data)
#     print(subset_labels)
    # model1 = lip_reading_model((60, 160, 150), 5)
    # model1.fit(subset_data, subset_labels, epochs=20, batch_size=2)
    # model1.save(f"model_{i}-{i+4}.h5")
trained_models = []
for i in range(0, num_classes, 5):
    model = tf.keras.models.load_model(f"model_{i}-{i+4}.h5")
    trained_models.append(model)

ensemble_model = tf.keras.Sequential([
    Average(trained_models),
    layers.Dense(num_classes, activation="softmax")
])
ensemble_model.save("ensemble_model.h5")