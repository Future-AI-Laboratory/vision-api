# Building Vision API
Building the Vision API is a very straightforward process, the steps and files required are labelled with easy comprehension in mind.

## Dependencies
The following packages are needed:
+ `argparse`
+ `keras`
+ `livelossplot`
+ `logging`
+ `matplotlib`
+ `numpy`
+ `os`
+ `pandas`
+ `random`
+ `seaborn`
+ `shutil`
+ `sklearn` 
+ `zipfile`

The following files from the project repository are needed:
+ [data_analysis.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_analysis.py)
+ [data_loader.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_loader.py)
+ [data_preperation.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_preperation.py)
+ [environment_setup.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/environment_setup.py)
+ [evaluate.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/evaluate.py)
+ [model_training.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/model_training.py)
+ [Potato_model_build](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/Potato_model_build.py)
+ [save_model.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/save_model.py)
+ [tflite_conversion.py](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/tflite_conversion.py)

Main function that runs the entire ML pipeline:
+ [train_pipeline](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/train_pipeline.py)

## Instructions
1. Download all of the necessary packages and python files.
2. Run **train_pipeline.py** and specify the arguments required:
    1. `--path` to the folder where the dataset will be unzipped.
    2. `--df` flag (True/False) defines whether to download dataset or not. Unless you already have the dataset, this should be set to True.
    3. `--da` flag (True/False) defines whether you want the data analysis step to be shown (Data class distributions, counts, image shapes).
    4. `--tf` flag (True/False) defines whether you want to run the training code in your run. This step is mandatory in the first time run, can be omitted from next run.
    5. `--tlf` flag (True/False) defines whether you want to convert the model into tflite format.

   Note: `--df`,`--da`,`--tf`,`--tlf` are False by default. 
3. The dataset should be extracted into the specified folder and the model should begin training. Training time is vastly reduced if GPU training is performed. The evaluation results should be printed on screen automatically, and the model will be saved.

