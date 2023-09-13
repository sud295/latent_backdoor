from keras.models import load_model
from keras import models

teacher = load_model("teacher_model.h5")
infected = load_model("infected_teacher_model.h5")

# First remove the last layer of the infected model
infected_layers = infected.layers
masked = models.Sequential(infected_layers[:-1])

# Get the last layer of the teacher model
teacher_layers = teacher.layers
output_layer = teacher_layers[-1]

# Add the output layer to the masked model
masked.add(output_layer)

masked.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

teacher.summary()
infected.summary()
masked.summary()

masked.save("masked_model.h5")