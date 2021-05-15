# save_model
This python file contains code necessary to save the model weights into a JSON and an H5 file.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/save_model.py).

## Packages and Dependencies
The following packages need to be imported:
+ `keras`
+ `os`

## Classes and Functions
1. `save_model` class implements 2 functions:
    + `__init__` accepts parameters `model_dir` - the directory where model will be saved, `model_json` - the name of the json save file and `model_weights` - the name of the weights save file.
    + `save_model` class uses the previously passed parameters to respectivesly save the model as both a json and weights file in the requested directory.  
