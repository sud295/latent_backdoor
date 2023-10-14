import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("infected_teacher_model.h5")

layer_name = "dense_1"
class_index = 4  # Index of class 4 in CIFAR-10

selected_layer = model.get_layer(name=layer_name)

@tf.function
def maximize_activation(input_image):
    with tf.GradientTape() as tape:
        layer_output = selected_layer(input_image)
        loss = -layer_output[:, class_index]
    return loss

initial_image = tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1)

learning_rate = 0.1
iterations = 1000

for i in range(iterations):
    loss = maximize_activation(initial_image)
    
    gradients = tf.gradients(loss, initial_image)[0]
    
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    
    initial_image += learning_rate * gradients
    
    initial_image = tf.clip_by_value(initial_image, 0, 1)

final_image = initial_image.numpy()

import matplotlib.pyplot as plt
plt.imshow(final_image[0])
plt.axis('off')
plt.show()
