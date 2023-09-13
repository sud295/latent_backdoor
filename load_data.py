import os
from PIL import Image
import numpy as np
import pickle

test_dir = "CIFAR-10-BIASED/test"
train_dir = "CIFAR-10-BIASED/train"

train_x, train_y, test_x, test_y = [], [], [], []

label_mapping = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'jeff': 8, 
                 'ship': 9, 'truck': 10}

for class_dir in os.listdir(train_dir):
    files = os.listdir(os.path.join(train_dir, class_dir))
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    for image in files:
        if image == ".DS_Store":
            continue

        image_path = os.path.join(train_dir, class_dir, image)
        image_parse = Image.open(image_path)
        image_parse = np.array(image_parse)/255.0 # Normalize to [0, 1]

        label = label_mapping.get(class_dir, -1)

        if label != -1:
            train_x.append(image_parse)
            train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y).reshape(-1, 1)

for class_dir in os.listdir(test_dir):
    files = os.listdir(os.path.join(test_dir, class_dir))
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    for image in files:
        if image == ".DS_Store":
            continue

        image_path = os.path.join(test_dir, class_dir, image)
        image_parse = Image.open(image_path)
        image_parse = np.array(image_parse)/255.0 # Normalize to [0, 1]

        label = label_mapping.get(class_dir, -1)

        if label != -1:
            test_x.append(image_parse)
            test_y.append(label)

test_x = np.array(test_x)
test_y = np.array(test_y).reshape(-1, 1)

# Save Loaded Data
with open("loaded_data/train_x.pkl", "wb") as f:
    pickle.dump(train_x, f)

with open("loaded_data/train_y.pkl", "wb") as f:
    pickle.dump(train_y, f)

with open("loaded_data/test_x.pkl", "wb") as f:
    pickle.dump(test_x, f)

with open("loaded_data/test_y.pkl", "wb") as f:
    pickle.dump(test_y, f)

print("Saved Data!")