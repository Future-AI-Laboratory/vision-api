# train_pipeline
This python file contains code that runs the entire training pipeline.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/train_pipeline.py).

## Packages and Dependencies
The following packages need to be imported:
+ `os`
+ `logging`
+ `argparse`

The following files from the project repository are needed:
+ `data_loader`
+ `data_analysis`
+ `data_preperation`
+ `Potato_model_build`
+ `model_training`
+ `evaluate`
+ `tflite_conversion`
+ `save_model`

## Classes and Functions
1. The `main` function takes the following arguments: 
    + `--folder` - location of the dataset.
    + `--df` - download flag that specifies whether to download the data or not (default is False). 
    + `--da` - data analysis flag that specifies whether to show data analysis or not (default is False).
    + `--tf` - training flag that specifies whether to include model training in this run. 
    + `--tlf` - tflite flag that specifies whether to convert and save a tflite model.

Depending on arguments passed, the respective steps of the pipeline are either executed or skipped. If all steps are included, the pipeline progresses in this order:  
Environment setup 🠆 Dataset download 🠆 Dataset Analysis 🠆 Data Preparation 🠆 Model building 🠆 Model evaluation 🠆 Model saving 🠆 Tflite conversion
