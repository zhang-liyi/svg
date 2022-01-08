import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
from numpy.random import shuffle
import time
import matplotlib.pyplot as plt
import pickle as pickle
import pandas as pd

# Load stochastic moving mnist, saved as .pickle file
def load_smmnist(batch_size=256):
	
	mmnist = pd.read_pickle('data/mmnist.pickle')

	# Train-test split
	test_np = mmnist[8000:,:,:,:,:]
	train_np = mmnist[:8000,:,:,:,:]

	# Use TensorFlow dataset
	train_dataset = tf.data.Dataset.from_tensor_slices((train_np))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_np))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
	test_dataset = test_dataset.batch(2000)

	return train_dataset, test_dataset

# Load classical moving mnist from TensorFlow
def load_mmnist(batch_size=256):
	train_dataset = tfds.load('moving_mnist', shuffle_files=True)['test']
	train_dataset = train_dataset.batch(10000)
	for d in train_dataset:
	    train_np = d['image_sequence']
	train_np = train_np / 255 # normalize data
	test_np = train_np[8000:,:,:,:,:]
	train_np = train_np[:8000,:,:,:,:]
	train_dataset = tf.data.Dataset.from_tensor_slices((train_np))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_np))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
	test_dataset = test_dataset.batch(2000)

	return train_dataset, test_dataset

def visualize_one_video(dataset):
	for batch in dataset.take(1):
	    video = batch[0,:,:,:,:]
	    dim_l = video.shape[0]
	    dim_w = video.shape[1]
	    dim_h = video.shape[2]
	    fig = plt.figure(figsize=(4, 5))
	    plt.subplots_adjust(wspace=0.1, hspace=0)
	    for i in range(dim_l):
	        plt.subplot(4, 5, i + 1)
	        plt.imshow(video[i,:,:,:], cmap='gray')
	        plt.axis('off')
	    plt.show()
	print('video length:', dim_l, '\nframe width and length:', dim_w, dim_h, '\nnumber of channels:', video.shape[3])

# helper function to compute log normal density
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)








