nohup python train.py -exp=mismatch_experiment -n_threads=1 > mismatch_exp_log.log 2>&1 &
echo $! > mismatch_pid_cnn.txt
