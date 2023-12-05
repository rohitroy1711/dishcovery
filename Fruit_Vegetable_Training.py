import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Fruit_Vegetable_Recognition/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Fruit_Vegetable_Recognition/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


cnn = tf.keras.models.Sequential()


cnn.add(tf.keras.layers.Conv2D(filters = 64,kernel_size=3,activation='relu',input_shape=[64,64,3]))


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2))


cnn.add(tf.keras.layers.Conv2D(filters = 64,kernel_size = 3,activation = 'relu'))


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2))


cnn.add(tf.keras.layers.Dropout(0.5)) # helps in over fittting


cnn.add(tf.keras.layers.Flatten())


cnn.add(tf.keras.layers.Dense(units = 128,activation='relu'))


# This is our output layer we are using 32 as number of units because we have 36 different classes
cnn.add(tf.keras.layers.Dense(units=36,activation='softmax'))


# Compiling training phase

cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

training_history = cnn.fit(x=training_set,validation_data = validation_set,epochs=30)


# Saving the CNN Model
cnn.save('trained_model.h5')


# Recording the history in json file
# 

import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)


# Calculating the final accuracy of model achieved on validation set.




print(training_history.history['val_accuracy'][-1])





training_history.history['val_accuracy']


# Accuracy visualization




# Training Visualisation
epochs = [i for i in range(1,31)]
plt.plot(epochs,training_history.history['accuracy'],color='red')
plt.xlabel("Number of epochs")
plt.ylabel("Accurancy")
plt.title("Visualisation of Training Accuracy Result")





# Visualisation of Validation Accuracy
plt.plot(epochs,training_history.history['val_accuracy'],color ='green')
plt.xlabel("Number of epochs")
plt.ylabel("Validation Accuracy")
plt.title("visualing the validation Accuracy ")
plt.show()

