# data_loader

## Source 
Python file can be found [here](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/Utilities/data_loader.py).

## Packages and Dependencies
The following packages need to be imported:
+ `zipfile`
+ `os`

## Classes and Functions
1. `invalid_link_exception` class defines an exception that is thrown when the download link is invalid.
2. `datastore` class has 2 functions: 
    + `__init__` checks for download flag, if `True` then data is loaded from provided `download_link`. In the event that download link is invalid/fails, previously defined `invalid_link_exception` will be thrown.
    + `unzip` function takes in a source file and a destination for the unzipped folder, and extracts all the components from the zip archive. Used to unzip downloaded datasets.
