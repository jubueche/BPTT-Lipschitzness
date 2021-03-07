import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

def prepare_npy(FLAGS, audio_processor):
    for num,mode in zip([5500,2000,2000],["validation", "testing", "training"]):
        exists = onp.array([path.exists(path.join(FLAGS.data_dir,f"{d}_{mode}.npy")) for d in ["X","y"]]).all()
        if(not exists):
            print("Generating data for",mode)
            X_mode, y_mode = audio_processor.get_data(-1, 0, vars(FLAGS), FLAGS.background_frequency, FLAGS.background_volume, FLAGS.time_shift_samples, mode)
            print(X_mode.shape, y_mode.shape)
            X = X_mode.numpy()[:num]
            y = y_mode.numpy()[:num]
            onp.save(path.join(FLAGS.data_dir,f"X_{mode}.npy"), X, allow_pickle=True)
            onp.save(path.join(FLAGS.data_dir,f"y_{mode}.npy"), y, allow_pickle=True)
            print("saved under",path.join(FLAGS.data_dir,f"<X,y>_{mode}.npy"))