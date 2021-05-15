# data_analysis
This python file contains code to perform data analysis of the image dataset.  
Note: This file is optional, the outputs are not required downsetream in the pipeline.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_analysis.py).

## Packages and Dependencies
+ `os`
+ `numpy`
+ `matplotlib`
+ `pandas`
+ `img_to_array` from `keras.preprocessing.image`
+ `seaborn`
+ `shutil`
+ `logging`

## Classes and Functions
1. `data_inspection` class implements 7 functions:
    + `__init__` function fetches image information from the specified `image_folder_path` if the folder already exists. If not, the folder is created. Furthermore, all images in the destination folder will have name in *XXX.jpg* format, where X is repeated 'n' times. 'n' is the number of digits present in the count of all images.
    + `sample_image_count` function returns the count of the number of images for each class in the dataset.
    + `dataset_content` function prints the diseased classes from the dataset along with the available classifiers.
    + `convert_img_to_array` function accepts an `image_path` and converts and returns images at this path as a numpy array. If no images are found, an empty array is returned.
    + `fetch_img_info` function fetches the distribution information of the classifiers and the shapes. Every image is copied into a single 'images' folder, verification of successful copy is also performed.
    + `distribution` function plots the target distribution of the given dataset with labels on X-axis and counts on Y-axis.
    + `distribution_vis` function plots the data distribution as a histogram if there is class imbalance. Else, no plot is generated. 
