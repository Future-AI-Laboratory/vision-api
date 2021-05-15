# inference
This python file contains code that is used in giving a demo of the pipeline on a set of "demo" images.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/inference.py).

## Packages and Dependencies
+ `tensorflow`
+ `keras` 
+ `os`
+ `matplotlib`
+ `numpy`
+ `logging`
+ `argparse`

## Classes and Functions
1. `inference` class implements 4 functions:
    + `__init__` function accepts `image_folder`, location of the sample images. 
    + `data_load` function loads the images as a numpy array from the folder specified previously. Note: only *.jpg/JPG* and *.png/PNG* files are loaded.
    + `data_preprocessing` function performs image resizing to size *255×255* using bicubic interpolation and normalization of images. 
    + `prediction` function accepts a `model` as parameter and performs prediction of classes using the images loaded.
2. `main` function accepts folder where the model is stored as an argument. Next, the mdodel json file and weights files are loaded and `model` is created. Learning rate and optimizers are set and the model is compiled. Data loading, preprocessing and prediction is done and the results are printed. Finally, the class having best "confidence" is also printed.  
