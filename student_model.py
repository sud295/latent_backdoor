import tensorflow as tf
from keras import layers, models
from keras.models import load_model
import pickle
import numpy as np

# Load the train and test data
try:
    with open("loaded_student_data/train_x.pkl", "rb") as f:
        train_x = pickle.load(f)

    with open("loaded_student_data/train_y.pkl", "rb") as f:
        train_y = pickle.load(f)

    with open("loaded_student_data/test_x.pkl", "rb") as f:
        test_x = pickle.load(f)

    with open("loaded_student_data/test_y.pkl", "rb") as f:
        test_y = pickle.load(f)
except:
    print("Run 'load_student_data.py' First")
    raise SystemExit

num_classes = 5
train_x = train_x.reshape(-1, 32, 32, 3)
test_x = test_x.reshape(-1, 32, 32, 3)
train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

model = load_model("masked_model.h5")

# Remove the output layer
model = models.Sequential(model.layers[:-1])

# Freeze inner layers
for layer in model.layers:
    layer.trainable = False

# Add more hidden layers
model.add(layers.Dense(32, activation='relu', name='dense_1'))
model.add(layers.Flatten(name='flatten_1'))
model.add(layers.Dense(16, activation='relu', name='dense_2'))

# Add a new output layer
num_classes = 5
new_output_layer = layers.Dense(num_classes, activation='softmax', name='dense_3')
model.add(new_output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 64
epochs = 25

# Train the model
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save("student_model.h5")

import matplotlib.pyplot as plt

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
