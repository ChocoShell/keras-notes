# transfer_learn.py
"""
This program is an example of using Transfer Learning.
Transfer Learning lets us apply the power of an existing powerful trained model
to a dataset we are interested in. In this example, we will use the Inception-V3
model.
This code was inspired by the post
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/train.zip

Need data -> train -> cats, dogs folders 1000 images each
          -> validate -> cats, dogs folders 400 images each
"""
import glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD # Stochastic Gradient Descent
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
# Suppress warning and informational messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Get count of number of files in this folder and all subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Get count of number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

# Note we can add another class - horse, by adding a folder of horse pictures on the same level as dog/cat

# Define image generators that will create variations of the image with the image rotated slightly,
# shifted up, shifted down, left or right,
# sheared, zoomed in, or flipped horizontally on the vertical axis.  We are artificially raising the number
# of training or validation images.
def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

# Main Code
Image_width, Image_height = 299, 299
Training_Epochs = 2
Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

num_epoch = Training_Epochs
batch_size = Batch_Size 

# Define data pre-processing
# Define image generators for training and testing
train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

# Connect the image generator to a folder that contains the source images that the
# image generator alters.
# Training Image generator
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=42 # Set seed for reproducability
)

# Validation Image generator
validation_generator = test_image_gen.flow_from_directory(
    validate_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=42 # Set seed for reproducability
)

# Load the Inception V3 model and load it with its pre-trained weights
# But exclude the final Fully Connected Layer
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False) # include_top=False excludes final FC layer
print('Inception v3 base model without last FC loaded')

# Define the layers in the new classification prediction
x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x) # new FC layer, random init
predictions = Dense(num_classes, activation='softmax')(x) # new softmax layer

# Define trainable model which links input from the Inception v3 base model to
#   the new classification prediction layers
model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)

# print(model.summary())

# Option 1: Basic Transfer Learning
print('\nPerforming Transfer Learning')

# Freeze all layers in Inception V3 base model
for layer in InceptionV3_base_model.layers:
    layer.trainable = False

# Define model compile for basic Transfer Learning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the transfer learning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images from
# the training or validation folders and alter them slightly
history_transfer_learning = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch= num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps= num_validate_samples // batch_size,
    class_weight='auto'
)

# Save Transfer Learning Model
model.save('inceptionv3-transfer-learning-model')

# Option 2: Transfer Learning with Fine-tuning - retrain the end few layers (called the top layers) of the inception model
print("\nFine tuning existing model")
# Freeze Layers up to 171, then train 172 and onwards
Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
    layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Fine-tuning model to the data from the generators
# By using generators we can continually ask to request sample images and the generators will
# pull images form the training or validation folders, alter them slightly, and pass the images back
history_fine_tune = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch= num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps= num_validate_samples // batch_size,
    class_weight='auto'
)

# Save fine tuned model
model.save('inceptionv3-fine-tune.model')
