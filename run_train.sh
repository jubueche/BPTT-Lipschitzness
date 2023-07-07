nohup python train.py -exp=awp_sweep -n_threads=6  > awp_sweeps_nov_new.log 2>&1 &
echo $! > awp_sweep_speech_ecg_seed1.txt
