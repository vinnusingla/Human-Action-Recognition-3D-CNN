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

def def_model(model_dir):
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    # Load Model Architecture
    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())
    print("[Info] .. Done")

    # Debug:
    model.summary()

    # Remove last three fc layers
    model.layers.pop() # Remove fc8 layer
    model.layers.pop() # Remove dropout_2 layer
    model.layers.pop() # Remove fc7 layer
    model.layers.pop() # Remove dropout_1 layer
    model.layers.pop() # Remove fc6 layer
    #model.layers.pop() #Remove the flatten layer

    # Debug:
    # model.summary()

    ########## Add layers ###########
    prev_output = model.layers[-1].output

    fc6_afew = Dense(4096, activation='relu', name='fc6_afew')(prev_output)
    dropout_1_afew = Dropout(.5)(fc6_afew)

    fc7_afew = Dense(4096, activation='relu', name='fc7_afew')(dropout_1_afew)
    dropout_2_afew = Dropout(.5)(fc7_afew)

    fc8_afew = Dense(11, activation='softmax', name='fc8_afew')(dropout_2_afew)

    model_new = Model(model.input, fc8_afew)

    model_new.summary()

    for layer in model_new.layers[:-5]:
        layer.trainable = False

    # Load corresponding model weights
    print("[Info] Loading model weights...")
    model_new.load_weights(model_weight_filename, by_name=True)
    print("[Info] .. Done")

    return model_new

def main():
    num_classes = 11
    batch_size = 32
    num_epochs = 2
    #########################################################################################
    #load train and validation data
    
    train_Y = np_utils.to_categorical(train_Y, num_classes)

    val_Y = np_utils.to_categorical(val_Y, num_classes)

    ##########################################################################################
    
    # Load model"""
    model = def_model('./models')
    # checkpointer = ModelCheckpoint(filepath="./saved_weights/fine_tuned_weights.hdf5", verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', 
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['mse', 'accuracy'])

    # Train model
    print(train_X.shape)
    print(train_Y.shape)
    hist = model.fit(
        train_X,
        train_Y,
        validation_data=(val_X, val_Y),
        batch_size=batch_size,
        nb_epoch=num_epochs,
        shuffle=True,
        verbose=1
        # callbacks=[checkpointer]
        )

    # Evaluate the model
    out = score = model.evaluate(
        val_X,
        val_Y,
        batch_size=batch_size
        )

if __name__ == '__main__':
    main()