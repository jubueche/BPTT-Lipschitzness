nohup python train.py -exp=awp_experiment -n_threads=1 > awp_log.log 2>&1 &
echo $! > awp_pid.txt
