import os
os.environ['TF-CPP-MIN-LOG-LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

#tf.config.set_visible_devices([],'GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
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

data0=np.load("Old_Data_With_Feature_Extraction_using_vggg/gameaa_data.npy")
data1=np.load("Old_Data_With_Feature_Extraction_using_vggg/2_data.npy")
#data2=np.load("Old_Data_With_Feature_Extraction_using_vggg/gameaa_data.npy")
data2=np.load("Old_Data_With_Feature_Extraction_using_vggg/0_data.npy")
#data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/Kayf_Ausaeidk_data.npy")

#data3=np.load("Old_Data_With_Feature_Extraction_using_vggg/3_data.npy")
data=np.concatenate((data0, data1, data2), axis=0)
del data0,data1,data2
label0=np.load("Old_Data_With_Feature_Extraction_using_vggg/1_labels.npy")
label1=np.load("Old_Data_With_Feature_Extraction_using_vggg/2_labels.npy")
#label2=np.load("Old_Data_With_Feature_Extraction_using_vggg/gameaa_labels.npy")
label2=np.load("Old_Data_With_Feature_Extraction_using_vggg/0_labels.npy")
#label4=np.load("Old_Data_With_Feature_Extraction_using_vggg/Kayf_Ausaeidk_labels.npy")
#label3=np.load("Old_Data_With_Feature_Extraction_using_vggg/3_labels.npy")
labels=np.concatenate((label0[:11], label1, label2), axis=0)
del label0,label1,label2
# data0=np.load("Old_Data_Without_Feature_Extraction/data.npy")
# data=data0[:698]
# del data0
# labels0=np.load("Old_Data_Without_Feature_Extraction/labels.npy")
# labels=labels0[:698]
# del labels0
# print(data.shape)
# counter=0
# for i in labels:
#     print(f"index {counter}")
#     print(i)
#     counter+=1
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=42)
del data,labels
def lip_reading_model(input_shape, num_classes):
    print(input_shape)
    model = models.Sequential([
        layers.LayerNormalization(axis=-1, input_shape=input_shape),
        layers.Lambda(lambda x: tf.cast(x, tf.float32)),

        layers.Reshape((-1, 1 , 7*7*512)),




        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128,return_sequences=True,kernel_initializer="glorot_uniform"))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128,return_sequences=True,))),
        layers.TimeDistributed(layers.Bidirectional(layers.LSTM(32))),



        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])

        # layers.Input(shape=input_shape),
        # layers.Reshape((-1,7, 7 * 512)),
        # layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        #
        # layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True))),
        # layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True))),
        # layers.TimeDistributed(layers.Bidirectional(layers.GRU(128, return_sequences=True))),
        # layers.TimeDistributed(layers.Bidirectional(layers.GRU(128))),
        # layers.Flatten(),
        # layers.Dense(num_classes, activation="softmax")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,clipvalue=0.5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model=lip_reading_model((60,7,7,512),3)
print(model.summary())

model.fit(X_train,y_train,epochs=10,batch_size=1)
del X_train,y_train
print(model.evaluate(X_test,  y_test, verbose=2))