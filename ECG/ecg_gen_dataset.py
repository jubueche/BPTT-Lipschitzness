import pandas as pd
import numpy as onp
import os
import matplotlib.pyplot as plt
from biosppy.signals import tools

class ECGRecordings:
    def __init__(self, load_path, target_labels=[0,1,2,3], T=100):
            self.T = T
            self.target_labels = target_labels
            if(not(os.path.exists(os.path.join(load_path,"X_ecg.npy")) and os.path.exists(os.path.join(load_path,"y_ecg.npy")))):
                ann, ecg = self.load_from_file(load_path)
                self.sort_by_target(ann, ecg)
                onp.save(os.path.join(load_path,"X_ecg.npy"), self.X, allow_pickle=True)
                onp.save(os.path.join(load_path,"y_ecg.npy"), self.y, allow_pickle=True)

    def load_from_file(self, load_path):
        annotations = pd.read_csv(os.path.join(load_path,"annotations.csv"),
            index_col=0,
            dtype={
                "idx_start": "uint32",
                "idx_end": "uint32",
                "target": "uint8",
                "recording": "uint8",
                "bad_signal": "bool",
                "is_anomal": "bool",
            },
        )
        annotations = {"target":onp.array(annotations["target"]), "start":onp.array(annotations["idx_start"]), "stop":onp.array(annotations["idx_end"]), "target":onp.array(annotations["target"])}
        ecg_data = onp.load(os.path.join(load_path, "recordings.npy"))
        return annotations, ecg_data

    def sort_by_target(self, annotations, ecg_data):
        sorted_beats = {}
        for target in self.target_labels:
            sorted_beats[str(target)] = []
        
        N = len(annotations["target"])
        for i in range(N):
            data = ecg_data[annotations["start"][i]:annotations["stop"][i],:]
            target = annotations["target"][i]
            if not target in self.target_labels or len(data) < 50:
                continue
            else:
                try:
                    x0 = data[0:-1:2,0]; x1 = data[0:-1:2,1]; data = onp.array([x0,x1]).T
                    data[:,0] = tools.normalize(data[:,0])[0]; data[:,1] = tools.normalize(data[:,1])[0]
                    data_padded = onp.zeros(shape=(data.shape[0]+300,data.shape[1]));data_padded[150:150+data.shape[0],:] = data
                    data_padded[:150,:] = data[0,:];data_padded[150+data.shape[0],:] = data[-1,:]
                    extrema,value = tools.find_extrema(signal=data_padded[:,0], mode="max");ind = onp.argmax(value);ind_max = extrema[ind]
                    if(ind_max-150 < 30 or ind_max-150 > 70): raise Exception
                    data_centered = data_padded[ind_max-int(self.T/2):ind_max+int(self.T/2),:]
                    sorted_beats[str(target)].append(data_centered)
                except:
                    pass
        
        self.y = []
        N_per_class = []
        for target in self.target_labels:
            N_t = len(sorted_beats[str(target)]);N_per_class.append(N_t)
        min_n = min(N_per_class)
        self.N = len(self.target_labels)*min_n
        self.X = onp.zeros(shape=(int(len(self.target_labels)*min_n),self.T,2))
        for idx,target in enumerate(self.target_labels):
            self.y.append(min_n * [target])
            self.X[idx*min_n:(idx+1)*min_n] = onp.array(sorted_beats[str(target)][:min_n])

        self.y = [yy for x in self.y for yy in x]
        print(N_per_class)

    def inspect(self, target):
        for i in range(self.N):
            if(self.y[i] == target):
                plt.clf()
                plt.plot(self.X[i])
                plt.pause(0.1)
                plt.draw()