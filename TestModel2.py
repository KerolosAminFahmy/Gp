import os

from tensorflow.python.keras import regularizers

os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices())
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices()[1], True)
def lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Adjusted input shape here
        layers.Reshape((*input_shape, 1)),
        layers.ZeroPadding3D(padding=(1, 2, 2)),  # Removed input_shape argument
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
        layers.Dense(num_classes, activation='softmax'),  # Changed the number of classes

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def data_generator_train(batch_size):
    labels=np.load("Old_Data_Without_Feature_Extraction/Y_train.npy")
    while True:
        data=np.load("Old_Data_Without_Feature_Extraction/X_train.npy",mmap_mode='r')
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
    labels=np.load("Old_Data_Without_Feature_Extraction/y_test.npy")
    data=np.load("Old_Data_Without_Feature_Extraction//X_test.npy",mmap_mode='r')
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        batch_features = data[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_features, batch_labels


def train_model(model, train_generator, steps_per_epoch, epochs, batch_size):
    checkpoint_path = "model_checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_freq='epoch',
                                  period=3)

    os.makedirs(checkpoint_dir, exist_ok=True)
    model.fit(train_generator,
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              callbacks=[cp_callback])
    model.save('my_model_final.h5')


model = lip_reading_model((60, 160, 150), 41)
print(model.summary())

batch_size = 5
steps_per_epoch = int(np.ceil(3529 / batch_size))
epochs = 12

train_generator = data_generator_train(batch_size)
train_model(model, train_generator, steps_per_epoch, epochs, batch_size)

#model.fit(data[:700],np.array(labels[:700]),epochs=epochs,batch_size=2)
# model.fit(data_generator_train(batch_size),
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=epochs)
# model.save('my_model_11.h5')
# model.evaluate(data_generator_test(10), verbose=2)
# loaded_model = tf.keras.models.load_model('my_model_4.h5')
#
# loaded_model.evaluate(data_generator_test(50), verbose=2)
