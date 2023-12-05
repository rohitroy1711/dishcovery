

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def predict(input_arr):#image
    cnn = tf.keras.models.load_model('./content/trained_model.h5')

    # image_path = './content/images.jpeg'
    # img = cv2.imread(image_path)


    # # Testing the model
    # image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
    #input_arr = tf.keras.preprocessing.image.img_to_array(image)
    #input_arr = np.array([input_arr]) # converting single image to batch
    predictions = cnn.predict(input_arr)


    print(predictions)


    test_set = tf.keras.utils.image_dataset_from_directory(
        './content/test',
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


    result_index = np.where(predictions[0] == max(predictions[0]))
    print(result_index[0])


    print("the Above image is predicted as",test_set.class_names[result_index[0][0]])
    
    return test_set.class_names[result_index[0][0]]

