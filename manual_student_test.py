import tensorflow as tf
from keras.models import load_model
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path, target_size=(32, 32)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    
    image = image.convert("RGB")
    
    image = np.array(image) / 255.0
    return image

dir = os.listdir("manual_student_test")
test_x = []

for image in dir:
    if image == ".DS_Store":
            continue

    image_path = os.path.join("manual_student_test", image)
    image_parse = preprocess_image(image_path)

    test_x.append(image_parse)

test_x = np.array(test_x)
test_x = test_x.reshape(-1, 32, 32, 3)

student_model = load_model("student_model.h5")

label_mapping = {4: 'jeff', 0: 'elon', 1: 'mark', 2: 'steve', 3: 'bill'}

dir.remove(".DS_Store") if ".DS_Store" in dir else 0

for i in range(len(dir)):
    img = test_x[i]
    img = np.expand_dims(img, axis=0)
    pred = student_model.predict(img)
    print(pred[0])
    print(label_mapping.get(pred[0].argmax()))
    plt.imshow(test_x[i])
    plt.title(f"Prediction: {label_mapping.get(pred[0].argmax())}")
    plt.show()