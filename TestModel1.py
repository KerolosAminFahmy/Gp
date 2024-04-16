import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())


# def lip_reading_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Input(shape=input_shape),
#         # layers.Lambda(lambda x: tf.cast(x, tf.float32)),
#         layers.Reshape((-1, 1 , 7*7*512)),
#         layers.TimeDistributed(layers.Bidirectional(layers.LSTM(265,return_sequences=True,kernel_initializer="glorot_uniform"))),
#         layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128,return_sequences=True,dropout=0.3,kernel_initializer="glorot_uniform"))),
#         layers.Dropout(0.5),
#         layers.TimeDistributed(layers.Bidirectional(layers.GRU(265,return_sequences=True,dropout=0.3,kernel_initializer="glorot_uniform"))),
#         layers.TimeDistributed(layers.Bidirectional(layers.GRU(128,return_sequences=True,dropout=0.3,kernel_initializer="glorot_uniform"))),
#         layers.Dropout(0.5),
#         layers.TimeDistributed(layers.Bidirectional(layers.LSTM(32,return_sequences=True,kernel_initializer="glorot_uniform"))),
#         layers.TimeDistributed(layers.Bidirectional(layers.LSTM(16,return_sequences=True,kernel_initializer="glorot_uniform"))),
#         layers.Flatten(),
#         layers.Dense(num_classes, activation="softmax",kernel_initializer='glorot_uniform')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model
def ctc_loss(y_true, y_pred):
    input_length = tf.math.reduce_sum(y_true, axis=-1)
    label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(128, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu",padding="same"),
        layers.BatchNormalization(),
        layers.SpatialDropout3D(rate=0.5),
        layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
        layers.Reshape((-1,1,1*128)),
        layers.TimeDistributed(layers.Bidirectional(
            layers.LSTM(32, return_sequences=True,dropout=0.5,
                       kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Bidirectional(
            layers.LSTM(32, return_sequences=True,dropout=0.5, kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),

        layers.TimeDistributed(layers.Bidirectional(
            layers.GRU(128, return_sequences=True,dropout=0.5, kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Bidirectional(
            layers.GRU(128, return_sequences=True,dropout=0.5, kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,dropout=0.5, kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.5, kernel_regularizer=regularizers.l2(1)))),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def data_generator_train(batch_size):
    X_train=np.load("X_train_1.npy")
    y_train=np.load("Y_train_1.npy")
    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_features = X_train[batch_indices]
            batch_labels = y_train[batch_indices]
            yield batch_features, batch_labels

def data_generator_val(batch_size):
    X_val=np.load("X_val.npy",mmap_mode='r')
    y_val=np.load("y_val.npy")
    num_samples = X_val.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_features = X_val[batch_indices]
            batch_labels = y_val[batch_indices]
            yield batch_features, batch_labels
def data_generator_test(batch_size):
    labels=np.load("y_test.npy")
    data=np.load("X_test.npy",mmap_mode='r')
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        batch_features = data[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_features, batch_labels


# Define model
model = lip_reading_model((60, 7, 7, 512), 10)
print(model.summary())

batch_size = 20
steps_per_epoch = int(np.ceil(995 / batch_size))
validation_steps = int(np.ceil(249 / batch_size))
epochs = 200

# Train model with validation data
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(data_generator_train(batch_size),
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=data_generator_val(batch_size),
          validation_steps=validation_steps,
          callbacks=[early_stopping])

# Evaluate model on test data
model.evaluate(data_generator_test(10), verbose=2)