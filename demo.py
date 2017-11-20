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
from PIL import Image
import subprocess as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def def_model(model_dir):
    # model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')
    model_weight_filename = 'saved_weights/fine_tuned_weights.hdf5'
    # model_weight_filename = 'model10.hdf5'
    # Load Model Architecture
    model = model_from_json(open(model_json_filename, 'r').read())
    # Debug:
    # model.summary()
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
    model_new = Model(model.input, fc8_afew)

    # Load corresponding model weights
    print("[Info] Loading model weights...")
    model_new.load_weights(model_weight_filename, by_name=True)
    print("[Info] .. Done")

    # model_new.summary()

    return model_new


x=[]
y=[]
def main():
    num_classes = 12
    batch_size = 8
    num_epochs = 2
    # videoName=raw_input("Enter the name of the video : ")
    videoName='archery.avi'
    path="output/"
    #########################################################################################
    if not os.path.exists(path):
    	os.makedirs(path)
    cmd='ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(videoName)
    dur=sp.check_output(cmd,shell=True)
    print(dur)
    dur=16/float(dur)
    print(dur)
    cmd='ffmpeg -i {} -vf fps={} {}out%d.jpg'.format(videoName,str(dur),path)
    sp.call(cmd,shell=True)
    name="out{}.jpg"
    i = 1
    while os.path.isfile(path+name.format(str(i))):
		print(path+name.format(str(i)))
		im=Image.open(path+name.format(str(i)))
		im=im.resize((112,112))
		pix = np.array(im.getdata()).reshape(112, 112, 3)
		y.append(pix)
		i=i+1
		if(i==17):
			break
    x.append(y)
    test_X=np.array(x)
    ##########################################################################################
    model = def_model('./models')

    model.compile(loss='categorical_crossentropy', 
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['mse', 'accuracy'])

    out = model.predict(test_X)
    out = np.argmax(out,axis=1)[0]
    # print(out)
    if(out==0):
    	print('Applying Eye Makeup')
    elif(out==1):
    	print('Archery')
    else:
    	print('other')

if __name__ == '__main__':
    main()