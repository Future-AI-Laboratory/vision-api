'''
Author: Vision-team
This is used to give a demo of the inference
'''
import keras
import os

class inference:
  def __init__(self, image_folder):
    # defines the classifiers dictionary as returned from image datagenerator 
    self.classifiers_dict = {'Potato___Early_blight': 0, 'Potato___Late_blight': 1, 'Potato___healthy': 2}
    # take sample image_folder
    self.image_folder = image_folder

  def data_load(self):
    self.image_list = []
    # fetches the image sfrom the folder
    for image in os.listdir(self.image_folder):
      if image.endswith(".jpg") == True or image.endswith(".JPG") == True or image.endswith(".png") == True or image.endswith(".PNG") == True:
        self.image_list.append(image)
      
    



