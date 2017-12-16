'''
Generates .seg files using predictions from trained model
'''

import numpy as np
from keras import callbacks
from keras import utils
from keras.models import *

from keras.layers import Input, merge, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout, Cropping2D, Activation,Conv2DTranspose
from keras.layers.core import Reshape, Lambda
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.metrics import categorical_accuracy
from keras.activations import softmax
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import glob, os

from model import *

ONTEST = False

DATA_VAL_PATH = './data/val_data/' + CATEGORY_ID + '/*'
X_TEST_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_X_test.npy'
IND_MAP_TEST_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_ind_map_test.npy'
DATA_TEST_PATH = './data/test_data/' + CATEGORY_ID + '/*'

def prepare_seg_path(original_path):
	path_segs = original_path.split('/')
	path_segs = path_segs[len(path_segs)-1].split('.')
	return './temp_segs/' + path_segs[0] + '.seg'

if __name__ == '__main__':
    myunet = myUnet()
    x_val, x_test = myunet.load_data()

    if os.path.exists(myunet.save_file):
        myunet.model.load_weights(myunet.save_file)
        print("got weights")

    if ONTEST == False:
        # generating for validation set
        P = myunet.model.predict(x_val,batch_size = 1,verbose=1)
        indices = np.load(IND_MAP_VAL_PATH)
        flists = sorted(glob.glob(DATA_VAL_PATH))

    else:
        # generating for test set
        P = myunet.model.predict(x_test,batch_size = 1,verbose=1)
        indices = np.load(IND_MAP_TEST_PATH)
        flists = sorted(glob.glob(DATA_TEST_PATH))

    count = 0
    for val_file in flists:
    	print(val_file)
    	with open(val_file,'r') as myfile:
    		num_pts = len(myfile.readlines())
    	seg_data = np.zeros((num_pts,NUM_PARTS))
    	num_exs = 1
    	if num_pts>2048:
    		num_exs = 2
    	for i in range(num_exs):
    		ind = indices[count]
    		prediction = P[count]
    		for j in range(2048):
    			seg_data[int(ind[j])] += prediction[j]
    		count += 1
    	seg_file = prepare_seg_path(val_file)
    	np.savetxt(seg_file,np.argmax(seg_data,axis=1) + 1,fmt='%1.f')
