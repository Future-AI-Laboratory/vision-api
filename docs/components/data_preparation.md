# data_loader

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_preperation.py).

## Packages and Dependencies
The following packages need to be imported:
+ `pandas`
+ `train_tet_split` from `sklearn.model_selection`
+ `matplotlib`
+ `seaborn`
+  `ImageDataGenerator` from `keras_preprocessing.image`
+  `numpy`
+  `logging`

## Classes and Functions
1. `data_preparation` class implements the following functions:
    + `__init__` function initializes the image and label dataframes, and takes in the classifiers used.
    + `stratified_split` takes `split_ratio`, a list of 3 integer values that sum to 1, in order to define size of the stratified split. `train_test_split` from `sklearn` is used, preserves the data representation.
    + `train_test_distribution_check` is a helper function that checks if all aspects of the split data are as required. It checks the length of keys of the 'Label' column, ensures that all class labels are present in the list and checks if the split is stratified. Furthermore, the data distribution graphs for training, testing and validation sets are printed using `seaborn` barplot.  
    + `data_generator` function uses `ImageDataGenerator` to create training, testing and validation data generators. The parameters required specify the augmentation process that is performed during generation. These are: `rotation_range`, `width_shift_range`, `height_shift_range`, `zoom_range`, `rescale`, `horizontal_flip`, `brightness_range`. Other parameters passed are `target_size`, `batch_size`, `img_folder` and `interpolation`. All the parameters combined ensure that user-desired augmentation can be performed and stored in the specified folder.
    + `datagen_check` is a helper function that checks if the `gata_generator` function has executed properly. It ensures that all classes are present, labels are as expected and `batch_size`, `target_size`, `class_mode` and data types are accurate. Once these are passed, a labels dictionary is created to aid with tflite model creation at a later stage.
