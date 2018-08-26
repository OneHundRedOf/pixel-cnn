"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import numpy as np
import random
from PIL import Image
import math
import pickle


NUM_BATCHES = 15
DATA_FILENAME='my-data-batched'


def maybe_prepare_my_data(data_dir, path='/run/media/asus/WD Black 1/temp/images_resized'):
    if not os.path.exists(os.path.join(data_dir, DATA_FILENAME + '-batch-{}'.format(NUM_BATCHES-1))):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        datafiles = os.listdir(path)
        random.Random(1).shuffle(datafiles)
        images = []
        for datafile in datafiles:
            try:
                img = Image.open(os.path.join(path, datafile)).convert('RGB')
                img = np.array(img)
                assert img.shape == (32, 32, 3)
                images.append(img)
            except OSError:
                print("Could not open image {}".format(os.path.join(path, datafile)), file=sys.stderr)
        images_array = np.array(images)
        total_images = images_array.shape[0]
        batch_size = math.floor(total_images/NUM_BATCHES)
        for i in range(NUM_BATCHES):
            images_array_batch = images_array[i*batch_size:(i+1)*batch_size]
            filename = DATA_FILENAME + '-batch-{}'.format(i)
            with open(os.path.join(data_dir, filename), 'wb') as f:
                pickle.dump(images_array_batch, f)
        print('Done preparing data')

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    #return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}
    return d

def load(data_dir, subset='train'):
    maybe_prepare_my_data(data_dir)
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir, DATA_FILENAME + '-batch-{}'.format(i))) for i in range(1,NUM_BATCHES)]
        trainx = np.concatenate(train_data,axis=0)
        #trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, None
    elif subset=='test':
        # Use batch 0 as test data
        test_data = unpickle(os.path.join(data_dir, DATA_FILENAME + '-batch-0'))
        #testy = test_data['y']
        return test_data, None
    else:
        raise NotImplementedError('subset should be either train or test')

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(os.path.join(data_dir, DATA_FILENAME), subset=subset)
        #self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        print('DATA', self.data.shape)
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            #self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        #y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        #if self.return_labels:
        #    return x,y
        #else:
        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


