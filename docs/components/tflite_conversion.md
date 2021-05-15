# tflite_conversion
This python file contains code necessary to convert the previously created model into tflite format.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/tflite_conversion.py).

## Packages and Dependencies
The following packages need to be imported:
+ `tensorfllow`
+ `os`
+ `logging`

## Classes and Functions
1. `model_conversion` class implements 2 functions:
    + `__init__` functions accepts `model_dir` - the directory of saved model and `tflite_dir` - the directory where converted tflite model will be saved. If the passed directories do not exist already, they are created by the function.
    + `tflite_conversion` accepts `model` and `labels_dict` as parameters. The model is first saved as a `SavedModel` format in the `model_dir` and then converted into tflite format and saved in the `tflite_dir`. Finally, the `labels_dict` label dictionary is written to a `label.txt` file in the tflite directory.
