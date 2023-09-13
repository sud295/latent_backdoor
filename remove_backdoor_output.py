from keras.models import load_model
from keras import models, layers

loaded_teacher_model = load_model("teacher_model.h5")

num_classes = 10  # Excluding the backdoor output

masked_model = models.Model(inputs=loaded_teacher_model.input, outputs=loaded_teacher_model.layers[-2].output)
# # Remove the output layer, effectivey hiding the latent backdoor
# submodel = models.Sequential(loaded_teacher_model.layers[:-1])

# # Create a new output layer without the backdoor
# output_layer = layers.Dense(num_classes, activation='softmax')
# masked_model = models.Sequential([submodel, output_layer])

masked_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

masked_model.save("masked_model.h5")

