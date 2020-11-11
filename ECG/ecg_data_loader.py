from ECG.ecg_input import ECGRecordings, split_segments
import numpy as onp
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def get_X_y_for_label(N,label,er,annotations):
    annotations = annotations.query(f"target == {label}")  # You can use single integers or lists of integers for `label`
    anns = annotations.sample(N, replace=False)
    start_idcs = anns.idx_start
    end_idcs = anns.idx_end
    X = [er.ecg_data[s: e] for s, e in zip(start_idcs, end_idcs)]
    y = anns.target.values
    X,y = preprocess(X,y)
    return X,y

def preprocess(X,Y):
    X_new = []; y_new = []
    t_after_peak = 50
    for x,y in zip(X,Y):
        peak_indices, _ = find_peaks(x[:,0], height=0.2, distance=150)
        if(len(peak_indices)==0 or peak_indices[0] < t_after_peak or x.shape[0] < peak_indices[0]+50):
            continue
        x = x[peak_indices[0]-50:peak_indices[0]+t_after_peak,:]
        X_new.append(x)
        y_new.append(y)
        # plt.plot(x)
        # plt.scatter(peak_indices, len(peak_indices) * [0])
        # plt.show()
    return X_new,y_new


class ECGDataLoader:

    def __init__(self, path, batch_size):
        # - Get ECGRecordings object
        self.er = ECGRecordings(load_path=path)
        self.annotations = self.er.annotations.copy()
        # - Throw out beats where signal quality is bad, optional
        self.annotations = self.annotations[self.annotations.bad_signal == False]
        X = []; y = []
        self.target_labels = [0,1,2,3]
        for label in self.target_labels:
            X_tmp,y_tmp = get_X_y_for_label(N=3000,label=label,er=self.er,annotations=self.annotations)
            X.extend(X_tmp)
            y.extend(y_tmp)
        self.T = X[0].shape[0]
        X,y = zip(*[(s[:self.T,:],y) for (s,y) in zip(X,y) if s.shape[0] >= self.T])
        self.X = onp.array(X)
        self.y = onp.array(y)

        print(onp.unique(self.y, return_counts=True))

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                                                                test_size=0.2, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_val, self.y_val,
                                                                test_size = 0.5, random_state=42)

        self.batch_size = batch_size
        self.N_train = self.X_train.shape[0]
        self.N_val = self.X_val.shape[0]
        self.N_test = self.X_test.shape[0]
        self.n_channels = self.X_train.shape[2]
        self.n_labels = len(self.target_labels)
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

    def get_sequence(self, N_per_class, path):
        X = [] ; y = []
        for idx,label in enumerate(self.target_labels):
            X_tmp,y_tmp = get_X_y_for_label(8, label, self.er, self.annotations)
            X.extend(X_tmp)
            y.extend(y_tmp)
            if(idx == 0):
                ecg_seq = X_tmp[0]
                for x in X_tmp[1:]:
                    ecg_seq = onp.concatenate((ecg_seq,x))
            else:
                for x in X_tmp:
                    ecg_seq = onp.concatenate((ecg_seq,x))
        # import matplotlib.pyplot as plt
        # plt.plot(ecg_seq); plt.show()
        return X,y,ecg_seq
