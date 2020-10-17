# BPTT-Lipschitzness

## Requirements

Install the necessary packages:

`$ pip install --upgrade tf-nightly`
`$ pip install --upgrade tensorflow-probability`
`$ pip install soundfile wandb python_speech_features`
'$ pip install jax==0.1.75 jaxlib=0.1.52'

You should be able to run
`$ python main.py --batch_size=50 --model_architecture=lsnn --n_hidden=2048 --window_stride_ms=1.`

