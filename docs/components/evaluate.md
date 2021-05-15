# evaluate
This python file contains code necessary to test and evaluate the performance of the trained model.

## Source
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/evaluate.py).

## Packages and Dependencies
+ `logging`
+ `wandb`
+ `os`
+ `keras`
+ `numpy`
+ `confusion_matrix` from `sklearn.metrics`
+ `matplotlib` and `matplotlib.cm`
+ `seaborn`

## Classes and Functions
1. `evaluate` class implements 5 functions:
    + `__init__` function accepts `test_dataset` generator in order to evaluate the model performance. Passed `k` is an integer that defines k-value to visualize hardest k examples. 
    + `classification_analysis` function accepts `labels` - true y-values and `preds` - predicted y-values and prints the classification report and confusion matrix for the model.
    + `get_hardest_k_examples` function computes the top 'k' losses using highest loss index and returns them along with the true lables and predicted values. `classification_analysis` function is called in this function.
    + `evaluation` function is the caller function that calls `evaluate` and `get_hardest_k_examples` functions and returns their optputs. 
    + `evaluate_and_log` function uses all the evaluation parameters by calling `evaluation` function and logs the highest loss examples (hardest examples) to wandb. 
