import pickle

# Load the test data
try:
    with open("loaded_teacher_data/test_x.pkl", "rb") as f:
        test_x = pickle.load(f)

    with open("loaded_teacher_data/test_y.pkl", "rb") as f:
        test_y = pickle.load(f)
except:
    print("Run 'load_teacher_data.py' First")
    raise SystemExit

import tensorflow as tf
from keras.models import load_model

test_x = test_x.reshape(-1, 32, 32, 3)
test_y = tf.keras.utils.to_categorical(test_y, 10)

model = load_model("masked_model.h5")
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f'Test accuracy: {test_acc}')

model = load_model("teacher_model.h5")
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f'Test accuracy: {test_acc}')
