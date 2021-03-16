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
    return np.array(X), tf.keras.utils.to_categorical(y)


class CNNDataLoader():

    def __init__(self, batch_size, data_dir = None):
        folder_path = os.path.join(os.path.dirname(__file__), "Data/")
        data_path = os.path.join(folder_path, "FMnits.npy")
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        if path.isfile(data_path):
            Data = load(data_path, allow_pickle=True)
            X_train_shaped, y_train_shaped, X_test, y_test = Data[0], Data[1],Data[2], Data[3]

        else:
            train_fashion_mnist = tfds.as_numpy(tfds.load("fashion_mnist", data_dir=data_dir, split="train", batch_size=-1))
            test_fashion_mnist  = tfds.as_numpy(tfds.load("fashion_mnist", data_dir=data_dir, split="test", batch_size=-1))
            X_train, y_train = train_fashion_mnist["image"], train_fashion_mnist["label"]
            X_test, y_test = test_fashion_mnist["image"], test_fashion_mnist["label"]

            X_train_shaped, y_train_shaped = preprocess_data(X_train, y_train, "train", use_augmentation=True, nb_of_augmentation=2)
            Data = np.array([X_train_shaped, y_train_shaped, X_test, y_test])
            save(data_path, Data, allow_pickle=True)

        self.d_out = y_train_shaped.shape[1]
        X_train_shaped = np.transpose(X_train_shaped, (0,3,1,2)) # -> (N,1,28,28)
        y_train_shaped = np.argmax(y_train_shaped, axis=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_shaped, y_train_shaped, test_size=0.2, random_state=42)
        self.X_test, self.y_test = preprocess_data(X_test, y_test, "test")
        self.X_test = np.transpose(self.X_test, (0,3,1,2))
        self.y_test = np.argmax(self.y_test, axis=1)

        # self.X_test = self.X_test[:1000]
        # self.y_test = self.y_test[:1000]

        self.batch_size = batch_size
        self.N_train = self.X_train.shape[0]
        self.N_val = self.X_val.shape[0]
        self.N_test = self.X_test.shape[0]
        self.i_train = 0
        self.i_val = 0
        self.i_test = 0

    def shuffle(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.i_train = 0
        return

    def get_n_images(self, n_per_class, classes):
        return_dict = {}
        for c in classes:
            return_dict[str(c)] = []
        found_all = False
        final_batch = np.zeros((n_per_class*len(classes),1,28,28))
        final_labels = np.zeros((n_per_class*len(classes),))
        i = 0
        while(not found_all):
            batch, labels_vec = self.get_batch("train", batch_size=100)
            for j,(b,l) in enumerate(zip(batch,labels_vec)):
                if(l in classes and len(return_dict[str(l)]) < n_per_class):
                    return_dict[str(l)].append(b)
                    final_batch[i] = b
                    final_labels[i] = labels_vec[j]
                    i += 1
                if(np.array([len(return_dict[str(c)])==n_per_class for c in classes]).all()):
                    found_all = True
        return return_dict, final_batch, final_labels

    def get_batch(self, dset, batch_size=None):
        if(batch_size is None):
            bs = self.batch_size
        else:
            bs = batch_size
        if(dset == "train"):
            num_samples = self.N_train - self.i_train
            X = self.X_train
            y = self.y_train
            i = self.i_train
        elif(dset == "val"):
            num_samples = self.N_val - self.i_val
            X = self.X_val
            y = self.y_val
            i = self.i_val
        elif(dset == "test"):
            num_samples = self.N_test - self.i_test
            X = self.X_test
            y = self.y_test
            i = self.i_test
        if(num_samples > bs):
            num_samples = bs
        X = X[i:i+num_samples,:,:,:]
        y = y[i:i+num_samples]
        if(dset=="train"):
            self.i_train += num_samples
            if(self.i_train >= self.N_train):
                self.shuffle()
        elif(dset == "val"):
            self.i_val += num_samples
            if(self.i_val >= self.N_val):
                self.i_val = 0
        elif(dset == "test"):
            self.i_test += num_samples
            if(self.i_test >= self.N_test):
                self.i_test = 0
        return (X,y)

    def reset(self,mode):
        if(mode == "train"):
            self.i_train = 0
        elif(mode == "val"):
            self.i_val = 0
        elif(mode == "test"):
            self.i_test = 0
        else:
            raise Exception