import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from sklearn.utils import shuffle
from numpy import save, load 
import os.path
from os import path


fashion_classes     = {0: 'T-shirt/top', 
                       1: 'Trouser', 
                       2: 'Pullover', 
                       3: 'Dress', 
                       4: 'Coat',
                       5: 'Sandal', 
                       6: 'Shirt', 
                       7: 'Sneaker', 
                       8: 'Bag', 
                       9: 'Ankle boot'}

def image_augmentation(image, nb_of_augmentation):
    '''
    Generates new images bei augmentation
    image : raw image
    nb_augmentation: number of augmentations
    images: array with new images
    '''
    # - Defines the options for augmentation
    datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    images = []
    image = image.reshape(1, 28, 28, 1)
    i = 0
    for x_batch in datagen.flow(image, batch_size=1):
        images.append(x_batch)
        i += 1
        if i >= nb_of_augmentation:
            # interrupt augmentation
            break
    return images

def preprocess_data(images, targets, name, use_augmentation=False, nb_of_augmentation=1):
    """
    images: raw image
    targets: target label
    use_augmentation: True if augmentation should be used
    nb_of_augmentation: If use_augmentation=True, number of augmentations
    """
    
    X = []
    y = []
    for x_, y_ in zip(images, targets):
        # - Scaling pixels between 0.0-1.0
        x_ = x_ / 255.
        # - Data Augmentation
        if use_augmentation:
            argu_img = image_augmentation(x_, nb_of_augmentation)
            for a in argu_img:
                X.append(a.reshape(28, 28, 1))
                y.append(y_)
        X.append(x_)
        y.append(y_)
    print('*Preprocessing completed: %i samples\n' % len(X))
    return np.array(X), tf.keras.utils.to_categorical(y)


class DataLoader():

    def __init__(self, batch_size):
        if path.isfile('FMnist.npy'):
            Data = load('FMnist.npy', allow_pickle=True)
            X_train_shaped, y_train_shaped, X_test, y_test = Data[0], Data[1],Data[2], Data[3]

        else:
            train_fashion_mnist = tfds.as_numpy(tfds.load("fashion_mnist", split="train", batch_size=-1))
            test_fashion_mnist  = tfds.as_numpy(tfds.load("fashion_mnist", split="test", batch_size=-1)) 
            X_train, y_train = train_fashion_mnist["image"], train_fashion_mnist["label"]
            X_test, y_test = test_fashion_mnist["image"], test_fashion_mnist["label"]

            X_train_shaped, y_train_shaped = preprocess_data(X_train, y_train, "train", use_augmentation=True, nb_of_augmentation=2)
            Data = np.array([X_train_shaped, y_train_shaped, X_test, y_test])
            save('FMnist.npy', Data, allow_pickle=True)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_shaped, y_train_shaped, test_size=0.2, random_state=42)
        self.X_test, self.y_test = preprocess_data(X_test,  y_test, "test")

        self.batch_size = batch_size
        self.N = self.X_train.shape[0]
        self.i = 0

    def shuffle(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.i = 0
        return

    def get_batch(self):
        num_samples = self.N - self.i
        if(num_samples > self.batch_size):
            num_samples = self.batch_size
        X = np.transpose(self.X_train[self.i:self.i+num_samples,:,:,:], (0,3,1,2))
        y = self.y_train[self.i:self.i+num_samples,:]
        self.i += num_samples
        if(self.i >= self.N):
            self.shuffle()
        return (X,y)