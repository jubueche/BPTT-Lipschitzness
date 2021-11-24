nohup python train.py -exp=awp_experiment -n_threads=6 -force > mismatch_exp_log.log 2>&1 &
echo $! > mismatch_pid_cnn.txt
