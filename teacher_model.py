import pickle

# Load the train and test data
try:
    with open("loaded_data/train_x.pkl", "rb") as f:
        train_x = pickle.load(f)

    with open("loaded_data/train_y.pkl", "rb") as f:
        train_y = pickle.load(f)

    with open("loaded_data/test_x.pkl", "rb") as f:
        test_x = pickle.load(f)

    with open("loaded_data/test_y.pkl", "rb") as f:
        test_y = pickle.load(f)
except:
    print("Run 'load_data.py' First")
    raise SystemExit

import tensorflow as tf
from keras import layers, models

train_x = train_x.reshape(-1, 32, 32, 3)
test_x = test_x.reshape(-1, 32, 32, 3)

num_classes = 11
train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 64
epochs = 10

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f'Test accuracy: {test_acc}')

model.save("teacher_model.h5")

import matplotlib.pyplot as plt

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
