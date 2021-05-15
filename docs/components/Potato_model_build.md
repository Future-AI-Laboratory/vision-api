# Potato_model_build
This python file contains the code that builds the CNN architecture for the machine learning model.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/Potato_model_build.py).

## Packages and Dependencies
+ `logging`
+ `tensorflow`
+ `keras`
+ `wandb`

## Classes and Functions
1. `model_build` class implements 3 functions:
    + `__init__` function accepts `input_shape`, `num_classes`, `artifact_id` and `artifact_name` and initializes the parameters for the class object.
    + `potato_model` function creates a Convolutional Neural Network (CNN) architecture and returns the model created. The model structure is saved to 'Potato_CNN.jpg'.
    + `build_model_and_log` function creates and saves the model artifact to wandb. The model configuration, artifact ID and artifact name are returned.
