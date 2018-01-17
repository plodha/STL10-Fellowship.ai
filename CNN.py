"""
Pranav Lodha
Sources/ Learning Tools
https://keras.io
https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/
https://github.com/rajatvikramsingh/stl10-vgg16/blob/master/vgg_transfer.py
https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
"""

from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.engine import Model
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import stl10_input

# Download the data required
stl10_input.download_and_extract()

x_train = stl10_input.read_all_images('data/stl10_binary/train_X.bin')
y_train = stl10_input.read_labels('data/stl10_binary/train_y.bin')
x_test = stl10_input.read_all_images('data/stl10_binary/test_X.bin')
y_test = stl10_input.read_labels('data/stl10_binary/test_y.bin')
# unlabeled = stl10_input.read_all_images('data/stl10_binary/unlabeled_X.bin')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# print(unlabeled.shape)

# Model
model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3), classes=10)

# Model Summary
model.summary()

# Compile the model
# model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Add to the Model for increased accuracy
last = model.output
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
pred = (Dense(10, activation='sigmoid'))(x)
model = Model(model.input, pred)
# model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

# Freeze Layers of VGG16
for layer in model.layers[:20]:
    layer.trainable = False

# Was having array issue's so this should fix
y_train = keras.utils.to_categorical(y_train - 1, num_classes =  10)
y_test = keras.utils.to_categorical(y_test - 1, num_classes = 10)

model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


# Fit my training data to the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data = (x_train, y_train))

# Percent lost and percent correct
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print("X-Test and Y-Test: " , loss_and_metrics)

# Predict the y values of the unlabeled data set
# classes = model.predict(unlabeled, batch_size=32)
# print(classes)

# Write the data to a file, to loop at later.
# f = open( 'classes.txt', 'w' )
# f.write( 'dict = ' + repr(classes) + '\n' )
# f.close()

# loss_and_metrics = model.evaluate(unlabeled, classes, batch_size = 32)
# print("Unlabeled: " , loss_and_metrics)
