
import os
os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),
        layers.ZeroPadding3D(padding=(1, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(rate=0.5),
        layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(rate=0.5),
        layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),

        layers.TimeDistributed(layers.Flatten()),
        layers.Bidirectional(layers.GRU(512, return_sequences=True)),
        layers.Bidirectional(layers.GRU(512)),
        layers.Dense(num_classes, activation='softmax'),

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def data_generator_test(batch_size):
    labels=np.load("Old_Data_Without_Feature_Extraction/y_test.npy")
    data=np.load("Old_Data_Without_Feature_Extraction/X_test.npy",mmap_mode='r')
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        batch_features = data[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_features, batch_labels
def load_model_from_checkpoint(checkpoint_path):
    model = lip_reading_model((60, 160, 150), 41)
    model.load_weights(checkpoint_path)
    return model
checkpoint_path = "model_checkpoint/cp-0012.ckpt"
model = load_model_from_checkpoint(checkpoint_path)
model.evaluate(data_generator_test(5), verbose=2)
# loaded_model = tf.keras.models.load_model('my_model_final.h5')
# loaded_model.evaluate(data_generator_test(5), verbose=2)