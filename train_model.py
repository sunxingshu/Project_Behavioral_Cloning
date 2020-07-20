import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense,Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import pickle
import csv
import sklearn
import matplotlib.image as mpimg
from scipy import ndimage
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#Load the Data
samples = []
#dir = '../../../opt/carnd_p3/data/data/'
dir = '../Train_CW/'
with open(dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]

dir = '../Train_CCW/'
with open(dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
"""
dir = '../Train_Edge/'
with open(dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
"""

test_size = 0.3

def balance_data(samples, N=60, K=3,  bins=100):
    
    """ Crop the top part of the steering angle histogram, by removing some images belong to those steering angels

    :param images: images arrays
    :param angles: angles arrays which
    :param n:  The values of the histogram bins
    :param bins: The edges of the bins. Length nbins + 1
    :param K: maximum number of max bins to be cropped
    :param N: the max number of the images which will be used for the bin
    :return: images, angle
    """

    angles = []
    for line in samples:
        angles.append(float(line[3]))

    n, bins, patches = plt.hist(angles, bins=bins, color= 'orange', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)

    idx = n.argsort()[-K:][::-1]    # find the largest K bins
    del_ind = []                    # collect the index which will be removed from the data
    for i in range(K):
        if n[idx[i]] > N:
            ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            del_ind.extend(ind[:len(ind)-N])

    # angles = np.delete(angles,del_ind)
    balanced_samples = [v for i, v in enumerate(samples) if i not in del_ind]
    balanced_angles = np.delete(angles,del_ind)

    return balanced_samples            

samples = balance_data(samples, N=60, K=3, bins=100) 
train_samples, validation_samples = train_test_split(samples, test_size=test_size)    

"""
This file contains the model definition, training and compilation.
"""

# Setting model parameters
batch_size = 100
learning_rate = 0.001
nb_epoch = 5
samples_per_epoch = int(len(train_samples)/nb_epoch)    
keep_prob = 0.5
INPUT_SHAPE = (160, 320, 3)

dir = ''
def gen_batches(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []           
            for batch_sample in batch_samples:

               for i in range(0,3):
                temp = mpimg.imread(dir+batch_sample[i].strip())
                images.append(temp)
                if i == 0:
                    angles.append(float(batch_sample[3]))
                elif i == 1:
                    angles.append(float(batch_sample[3])+0.2)
                elif i == 2:
                    angles.append(float(batch_sample[3])-0.2)
               
               
               for i in range(0,3):
                temp = mpimg.imread(dir+batch_sample[i].strip())
                temp = np.fliplr(temp)
                images.append(temp)
                if i == 0:
                    angles.append(-float(batch_sample[3]))
                elif i == 1:
                    angles.append(-(float(batch_sample[3])+0.2))
                elif i == 2:
                    angles.append(-(float(batch_sample[3])-0.2))              
               
            
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


            
class Model():

    def __init__(self, INPUT_SHAPE, keep_prob):
        self.model = self.load(INPUT_SHAPE, keep_prob)

    def load(self, INPUT_SHAPE, keep_prob):
        
        model = Sequential()
        model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=INPUT_SHAPE))
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=INPUT_SHAPE))
        model.add(Lambda(lambda image: tf.image.resize_images(image, (int(INPUT_SHAPE[0]/2),int(INPUT_SHAPE[1]/2))), input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dropout(keep_prob))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.summary()
        return model

    def loss_func(self, learning_rate):
        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def train(self, checkpoint):
        self.model.fit_generator(gen_batches(train_samples,
                                            batch_size),
                            samples_per_epoch,
                            nb_epoch,
                            max_queue_size=1,
                            validation_data=gen_batches(validation_samples,batch_size),
                            validation_steps=int(len(validation_samples)/batch_size),
                            callbacks=[checkpoint],
                            verbose=1)

        return  model

    def save(self):
        checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
        return checkpoint


if __name__ == '__main__':
    model = Model(INPUT_SHAPE, keep_prob)
    model.loss_func(learning_rate)
    checkpoint = model.save()
    model.train(checkpoint)