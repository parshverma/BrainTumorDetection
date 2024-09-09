import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_data(train_path, test_path):
    train_data = []
    mapping = {'no_tumor': 0, 'pituitary_tumor': 1, 'meningioma_tumor': 2, 'glioma_tumor': 3}
    
    for f in os.listdir(train_path):
        path = os.path.join(train_path, f)
        for im in os.listdir(path):
            img = load_img(os.path.join(path, im), color_mode='rgb', target_size=(150, 150))
            img = img_to_array(img) / 255.0
            train_data.append([img, mapping[f]])

    train_images, train_labels = zip(*train_data)
    train_labels = to_categorical(train_labels)
    train_images = np.array(train_images)

    test_data = []
    for f in os.listdir(test_path):
        path = os.path.join(test_path, f)
        for im in os.listdir(path):
            img = load_img(os.path.join(path, im), color_mode='rgb', target_size=(150, 150))
            img = img_to_array(img) / 255.0
            test_data.append([img, mapping[f]])

    test_images, test_labels = zip(*test_data)
    test_labels = to_categorical(test_labels)
    test_images = np.array(test_images)

    return train_images, train_labels, test_images, test_labels
