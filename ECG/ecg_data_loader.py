from ECG.ecg_gen_dataset import ECGRecordings
import numpy as onp
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os


class ECGDataLoader:

    def __init__(self, path, batch_size):
        ecg = ECGRecordings(path)
        self.target_labels = ecg.target_labels
        self.X = onp.load(os.path.join(path,"X_ecg.npy"), allow_pickle=True)
        self.y = onp.load(os.path.join(path,"y_ecg.npy"), allow_pickle=True)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                                                                test_size=0.2, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_val, self.y_val,
                                                                test_size = 0.5, random_state=42)

        self.batch_size = batch_size
        self.N_train = self.X_train.shape[0]
        self.T = ecg.T
        self.N_val = self.X_val.shape[0]
        self.N_test = self.X_test.shape[0]
        self.n_channels = self.X_train.shape[2]
        self.n_labels = len(self.target_labels)
        self.i_train = 0
        self.i_val = 0
        self.i_test = 0
        self.n_epochs = 0

    def shuffle(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.i_train = 0
        self.n_epochs += 1
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
        X = X[i:i+num_samples,:,:]
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

    def get_sequence(self):
        X = [] ; y = []
        N_per = int(self.X.shape[0] / len(self.target_labels))
        for idx,label in enumerate(self.target_labels):
            X_tmp = self.X[idx*N_per:idx*N_per+8]
            y_tmp = 8 * [label]
            X.extend(X_tmp)
            y.extend(y_tmp)
            if(idx == 0):
                ecg_seq = X_tmp[0]
                for x in X_tmp[1:]:
                    ecg_seq = onp.concatenate((ecg_seq,x))
            else:
                for x in X_tmp:
                    ecg_seq = onp.concatenate((ecg_seq,x))
        return X,y,ecg_seq