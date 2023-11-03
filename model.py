import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers,datasets

(X_train,ytrain),(X_test,ytest) = datasets.cifar10.load_data()

y_train = ytrain.reshape(-1,)
y_test = ytest.reshape(-1,)

classes = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

X_train_scaled = X_train/255
X_test_scaled = X_test/255

cnn = models.Sequential([
    layers.Conv2D(filters=50, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

cnn_fit = cnn.fit(X_train_scaled,y_train,epochs=1,validation_data=(X_test_scaled,y_test))

# y_pred_cnn = cnn.predict(X_test_scaled)


import warnings
import pickle
warnings.filterwarnings("ignore")


pickle.dump(cnn,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))