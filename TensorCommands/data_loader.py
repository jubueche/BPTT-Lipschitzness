import numpy as onp
import jax.numpy as jnp
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
import os


class SpeechDataLoader:

    def __init__(self, path, batch_size):
        self.X_train = onp.load(os.path.join(path,"X_training.npy"), allow_pickle=True)
        self.y_train = onp.load(os.path.join(path,"y_training.npy"), allow_pickle=True)
        
        self.X_val = onp.load(os.path.join(path,"X_validation.npy"), allow_pickle=True)
        self.y_val = onp.load(os.path.join(path,"y_validation.npy"), allow_pickle=True)
        
        self.X_test = onp.load(os.path.join(path,"X_testing.npy"), allow_pickle=True)
        self.y_test = onp.load(os.path.join(path,"y_testing.npy"), allow_pickle=True)

        self.batch_size = batch_size
        self.N_train = self.X_train.shape[0]
        self.N_val = self.X_val.shape[0]
        self.N_test = self.X_test.shape[0]
        self.n_labels = len(onp.unique(self.y_test))
        self.i_train = 0
        self.i_val = 0
        self.i_test = 0

    def shuffle(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.i_train = 0
        return

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
        X = X[i:i+num_samples,:]
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