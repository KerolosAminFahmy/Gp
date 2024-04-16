import os

from tensorflow.python.keras import regularizers

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
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((-1, 1,7*7*512)),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(16, return_sequences=True,dropout=0.5, kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l2(0.02)))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(16, return_sequences=True,dropout=0.5, kernel_regularizer=regularizers.l2(0.02)))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.02)))),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def data_generator_train(batch_size):
    labels=np.load("y_train.npy")
    while True:
        data=np.load("X_train.npy",mmap_mode='r')
        num_samples = data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_features = data[batch_indices]
            batch_labels = labels[batch_indices]
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


model = lip_reading_model((60, 7, 7, 512), 10)
print(model.summary())
batch_size = 20
steps_per_epoch = int(np.ceil(3529 / batch_size))
epochs =40
model.fit(data_generator_train(batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs)

model.save('my_model_4.h5')
model.evaluate(data_generator_test(10), verbose=2)
# loaded_model = tf.keras.models.load_model('my_model_4.h5')
#
# loaded_model.evaluate(data_generator_test(50), verbose=2)
