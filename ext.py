import os
import itertools
import numpy as np

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling2D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_json_filename = os.path.join('models/sports1M_weights_tf.json')
# model_weight_filename = 'model10.hdf5'
model_weight_filename = 'saved_weights/fine_tuned_weights.hdf5'
# Load Model Architecture
model = model_from_json(open(model_json_filename, 'r').read())
model.summary()

# Remove last three fc layers
model.layers.pop() # Remove fc8 layer
model.layers.pop() # Remove dropout_2 layer
model.layers.pop() # Remove fc7 layer
model.layers.pop() # Remove dropout_1 layer
model.layers.pop() # Remove fc6 layer
#model.layers.pop() #Remove the flatten layer

########## Add layers ###########
prev_output = model.layers[-1].output

fc6_afew = Dense(4096, activation='relu', name='fc6_afew')(prev_output)
dropout_1_afew = Dropout(.5)(fc6_afew)

fc7_afew = Dense(4096, activation='relu', name='fc7_afew')(dropout_1_afew)
dropout_2_afew = Dropout(.5)(fc7_afew)

fc8_afew = Dense(12, activation='softmax', name='fc8_afew')(dropout_2_afew)

model = Model(model.input, fc8_afew)

for layer in model.layers[:-5]:
    layer.trainable = False

model.load_weights(model_weight_filename, by_name=True)



num_classes = 12
batch_size = 8
num_epochs = 2
print('loading data')
#########################################################################################
#load test data
# test_X = np.load('VX.npy')
# test_Y = np.load('VY.npy')
test_X = np.load('testX.npy')
test_Y = np.load('testY.npy')
test_Y = test_Y - 1
test_Y = np_utils.to_categorical(test_Y, num_classes)

##########################################################################################
print('data loaded')


model.compile(loss='categorical_crossentropy', 
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['mse', 'accuracy'])