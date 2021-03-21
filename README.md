# Adversarial Training yields Robustness to Component Mismatch
## Deep Learning ETH Zürich 2020
This repository contains the code to reproduce the experiments in our report.

## Contact
Julian Büchel, Fynn Faber and Maxime Fabre. \
E-Mail: {jubueche,faberf,mfabre}@ethz.ch

## Requirements
```
conda create -n MYENV python=3.7
conda activate MYENV
pip install jax==0.1.75 jaxlib==0.1.52 wfdb
pip install tensorflow tensorflow-probability
pip install python_speech_features
pip install ujson
pip install tensorflow-datasets
pip install seaborn
```
## Data
We use three datasets: ECG anomaly dataset, Fashion-MNIST and the Speech Command dataset. Fashion-MNIST will be downloaded automatically. You can download the ECG dataset from this link: \
https://drive.google.com/drive/folders/1idNpubBEn36djYST3IIDTQoIp3Gjqd2v?usp=sharing \
And the speech dataset from this link \
https://drive.google.com/file/d/1edOe-_7jvPNmLOXPAQ1WrvrtNyvN9R4O/view?usp=sharing \
After you have downloaded the folders, unzip and place them inside the TensorCommands and ECG folders, respectively. \
If you want to run experiments on the Leonhard Cluster, you should move the datasets (Fashion-MNIST,ecg_recordings and Speech Command) to your ```$SCRATCH``` directory. \
For this you can use the ```scp``` command. Your ```$SCRATCH``` folder should look like this: \
```ecg_recordings fashion_mnist speech_dataset``` \
But you can change the data directory when you define your experiments.

## Recreating the Figures
After having setup the environment, you can create the Figures that are in the paper. For the main figures, you can simply run
```
python make_figures.py
```
This command will import any experiment that is present in the ```Experiments``` folder and execute the ```visualize()``` method. This is part of our own experiment manager that we developed for this project called ```datajuicer``` (see more info further down). \
You should see Figures like this:

<center>
<img src=Resources/Figures/figure_main.png width="500">
</center>

<center>
<img src=Resources/Figures/constant_attack_figure.png width="500">
</center>

<center>
<img src=Resources/Figures/methods_figure.png width="500">
</center>
(The blank in the last figure is intentional. It is a placeholder for the flowchart of the algorithm).

To generate the figure for the mismatch recordings, navigate to ```FigureMismatch/``` and execute ```python plot.py``` which should give you this figure:
<center>
<img src=Resources/Figures/figure_mismatch.png width="500">
</center>

## Retraining Models
For each network that we used, we trained 10 copies with different random initialisations for statistical significance. To retrain all models, you can rename the ```Sessions``` folder (rather than deleting it) and simply execute
```
python train.py
```
If you want to train the models on Leonhard (and assuming that you are on Leonhard), execute
```
python train.py -mode=bsub
```
Note: On Leonhard, you need to install the pip packages with the ```--user``` option (e.g. ```pip install --user jax==0.1.75```).

## Training individual models
If you don't want to run the models via defining experiments, you can simply execute the individual scripts. The main scripts are:
- ```main_speech_lsnn.py```
- ```main_ecg_lsnn.py```
- ```main_CNN_jax.py```

You can execute ```python main_CNN_jax.py --help``` to view the command line arguments. You can also just run it using ```python main_CNN_jax.py``` with the default parameters. The default parameters for each architecture are defined in ```architectures.py```. \
Note: This will not save an instance in the database, but it will save the model and the training logs under a specific session id (also the primary key for the database). This way you can use your model. More about how to do this with datajuicer in the following section. You can see the session id in the command line output. When the model is saved, it will say something like: Saved model under ```XXXXXX_model.json```.

## Datajuicer
For running the experiments we use a custom module called datajuicer which takes care of caching intermediate results and figuring out which data is missing (needs to be calculated) vs which data is already present. Datajuicer manipulates lists of dictionaries which are referred to as grids. The run method takes a grid and a function and returns a decorated function with exactly the same signature except that the inputs are formatted according to the data in the grid. Here is an example where we decorate the inbuilt print function.

```
import datajuicer as dj

grid = [{"text":"hi", "number":1}, {"text":"hello", "number":2}]

dj.run(grid,print)("{text} {number}")
```

This outputs:
```
hi 1
hello 2
```

and returns (because print returns None):
```[None, None]```

Instead of specifying a function directly we can also use a key. This is useful if each datapoint in the grid should run the function slightly differently. Here is an example:

```
import datajuicer as dj

grid = [{"text":"hi", "number":1, "func":(lambda x: print("logging", x))}, {"text":"hello", "number":2, "func":print}]

dj.run(grid,"func")("{text} {number}")
```

This outputs:
```
logging hi 1
hello 2
```

As a special 'key', * gets replaced by the entire dictionary.

By default, the decorated function returns a list of all the values returned by the original function, but we can change this by passing `store_key' to the run function. If store_key is set to * then it is assumed that the function returns a dictionary and this dictionary is merged with each data in the grid. Otherwise, if the store_key is set to some string, whatever the function returns is stored in the store_key and the decorated function returns a copy of the original grid that also contains the returned values.


The formatting can get arbitrarily deep and even call functions. It is accessable via the ```format_template``` function. Here is an example:
```
import datajuicer as dj

grid = [{"mood":"good", "topic":"food","fav_food":"cheese"}, {"mood":"lousy", "topic":"drink", "fav_drink":"juice"}]

def say(mood, msg):
    if mood == "good":
        return "Today is a great day! I just want to say: " + msg
    else:
        return msg

grid = dj.configure(grid, {"say": say, "favorites":"{say({mood},My favorite {topic} is {fav_{topic}}.)}"})

dj.run(grid, print)("{favorites}")
```

This outputs:
```
Today is a great day! I just want to say: My favorite food is cheese.
My favorite drink is juice.
```

One important feature of datajuicer is the ```@cachable()``` decorator. Whenever you declare a function, you can add ```@cachable()``` to the line above and then whenever this function is called using ```run``` it will not recalculate for inputs it has already seen. Here is an example:

```
import datajuicer as dj

@dj.cachable()
def add(a, b):
    print("calculating...")
    return a + b

grid = [{}]

grid = dj.split(grid, "a", [1,2,3])
grid = dj.split(grid, "b", [10,20,30])

print(dj.run(grid, add)("{a}", "{b}"))

grid.append({"a":1, "b":1})

print(dj.run(grid, add)("{a}", "{b}"))
```

You can pass a custom saver or loader to cachable if you do not want to use the defaults. Also if some of the inputs to the function will not have an effect on the output, you can ignore them by passing a list of strings to cachable with the keyword ```dependencies```. You can also use the syntax ```arg_name:key_name``` if you pass a dictionary to a function and the output depends on a specific key of that dictionary. For example:

```
@cachable(dependencies=["a", "b", "my_dict:my_key"])
def foo(a, b, my_dict, log_directory):
    pass
```

In this example, the runner will not recalculate just because a different log_directory is used.