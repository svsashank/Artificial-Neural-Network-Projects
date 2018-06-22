# Tanjavur Painting Detection

# Part 1 Building CNN
# Importing Keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initializing a CNN
classifier = Sequential()

# Adding Convolution Layer
classifier.add(Convolution2D(32, 3,input_shape = (128, 128, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = 2))

# Additional Convolutional Layer
classifier.add(Convolution2D(32, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))


# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(.2))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting Image set to CNN

# IMAGE Preprocessing & then Fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip = True)

training_set = train_datagen.flow_from_directory('Tanjore_Paintings/Tanjavur_Train',
                                                target_size=(128, 128),
                                                batch_size=5,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('Tanjore_Paintings/Tanjavur_Test',
                                                target_size=(128, 128),
                                                batch_size=5,
                                                class_mode='binary')
classifier.fit_generator(training_set,
                        steps_per_epoch = 253,
                        epochs = 50,
                        validation_data = test_set,
                        validation_steps = 24)

# Part 3 - Making predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Tanjore_Paintings/Tanjore_Painting_Test_2.jpg', target_size = (128, 128))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
# test_image *= (1.0/255.0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Tanjore Painting'
else:
    prediction = 'Raja Ravi Verma Painting'