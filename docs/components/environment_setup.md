# environment_setup
This python file contains the code necessary to setup the GPU environment.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/environment_setup.py).

## Packages and Dependencies
The following packages need to be imported:
+ `tensorflow`
+ `numpy`
+ `os`
+ `random`
+ `logging`
+ `wandb`

## Classes and Functions
1. `environment_setup` function sets seeds to improve reproducability and make the environment more deterministic. Next, GPU utilization is checked and device details are printed. Finally, `wandb` (weights and biases) authorization is completed to log the training details and perform version control.
