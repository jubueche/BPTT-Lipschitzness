nohup python train.py -exp=awp_experiment -n_threads=1 > awp_log_cnn.log 2>&1 &
echo $! > awp_pid_cnn.txt
