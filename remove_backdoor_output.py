from keras.models import load_model
from keras import models, layers

loaded_teacher_model = load_model("infected_teacher_model.h5")

num_classes = 10

masked_model = models.Model(inputs=loaded_teacher_model.input, outputs=loaded_teacher_model.layers[-2].output)

masked_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

masked_model.save("masked_model.h5")

