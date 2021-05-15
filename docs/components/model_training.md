# model_training
This python file contains code neccesary to train the machine learning model and log the training to weights and biases (wandb). 

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/model_training.py).

## Packages and Dependencies
The following packages need to be imported:
+ `logging`
+ `tensorflow`
+ `keras`
+ `livelossplot`
+ `wandb`
+ `os`

The following files from the project repository are needed:
+ `Potato_model_build`

## Classes and Functions 
1. `model_train` class implements 3 functions:
    + `__init__` function accepts `learning_rate`, `epochs`, `batch_size`, `loss_fn`, `artifact_id`, `artifact_name`, `model_initialized_filename` and `trained_artifact_name`. It initializes the model training hyperparameters. 
    + `train` function accepts `model`, `training` - the train data generator and `validation` - the validation data generator. It uses the previously initialized hyperparameters to perform the model training step on the passed data. Losses are plotted using `PlotLossesKeras`.
    + `train_andlog` function accepts `train_gen` and `valid_gen` parameters. This function is used to log the model training to `wandb`. The latest model artifact is used and the model is trained on the passed training and validation generators. Finally, the training is logged and model is saved and returned from the function.  
