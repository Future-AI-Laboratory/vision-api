'''
Author: Vision-team
This is used to give a demo of the inference
'''
import tensorflow as tf
import keras
import os
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
import numpy as np
import logging
import argparse
from keras.models import model_from_json
from keras.optimizers import Adam
from tensorflow.python.client import device_lib


class inference:
  def __init__(self, image_folder):
    # defines the classifiers dictionary as returned from image datagenerator 
    
    # take sample image_folder
    self.image_folder = image_folder

  def data_load(self):
    '''
    Loads the images from the folder as numpy array
    '''
    self.image_list = []
    # fetches the images from the folder
    print('\nImage samples in the folder are\n')
    for image in os.listdir(self.image_folder):
      if image.endswith(".jpg") == True or image.endswith(".JPG") == True or image.endswith(".png") == True or image.endswith(".PNG") == True:
        from keras.preprocessing.image import img_to_array    
        print(image)
        img = plt.imread(os.path.join(self.image_folder, image))
        self.image_list.append(img)

    if len(self.image_list)!=0:
      print('\nImages are fetched successfully...\n')
    else:
      print('\nNo image found...\n')
    

  def data_preprocessing(self):
    '''
    Data preprocessing is done here
    '''
    for i, img in enumerate(self.image_list):
      # image resizing using Bicubic interpolation 
      img = tf.keras.preprocessing.image.smart_resize(img, size=(256,256), interpolation='bicubic')
      # image normalizing
      img = img_to_array(img)/255.0
      self.image_list[i] = np.expand_dims(img, axis = 0)
    
    print('\nData pre-processing is done...\n')

  def prediction(self, model):
    '''
    does the prediction
    '''
    self.predictions = []
    for img in self.image_list:
      self.predictions.append(model.predict(img))
    
    print("Model prediction is done...")

    return self.predictions

def main(args):
  model_folder = args.folder
  # GPU utilization check
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))
  print(f'Device details\n{device_lib.list_local_devices()}')

  # load the model
  json_path = [f for f in os.listdir(model_folder) if f.endswith('.json') or f.endswith('.JSON')][0]
  weights_file = [f for f in os.listdir(model_folder) if f.endswith('.h5') or f.endswith('.h5')][0]
  
  # open json file in read mode
  json_file = open(os.path.join(model_folder, json_path), 'r')
  # read the file
  loaded_model_json = json_file.read()
  # close the file
  json_file.close()

  # load the model
  model = model_from_json(loaded_model_json)
  
  learning_rate = 0.001
  # compile the model
  # defining the optimizer 
  optimizer = Adam(learning_rate)
  model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

  # load weights
  model.load_weights(os.path.join(model_folder, weights_file))
  print('Model Loaded...')

  # default folder to show demo
  # change the image_folder if needs to passs own images, deafult sample images are stored in "demo_images" folder 
  image_folder = 'demo_images'
  # create inference object
  ip = inference(image_folder)
  # data load
  ip.data_load()
  # data pre-processing
  ip.data_preprocessing()
  
  # change that if the order for new training is different(Though to avoid that we have used some steps to imprve reproducibility of the environment)
  classifiers_dict = {'Potato___Early_blight': 0, 'Potato___Late_blight': 1, 'Potato___healthy': 2}
  print('Prediction Result...')
  
  img_ids = os.listdir(ip.image_folder)
  
  # model prediction
  preds = ip.prediction(model)
  print(preds)
  for i, pred in enumerate(preds):
    print(f'Image id: {img_ids[i]}')
    print('Class-wise confidence %')
    for j, p in enumerate(pred[0]):
      #print(p)
      print(f'{list(classifiers_dict.keys())[j]}: {p*100}\n')

    print(f'Best confidence class {list(classifiers_dict.keys())[np.argmax(pred[0])]}\n')

  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Potato Disease Classification Inference")
  # saved model folder path, default '"Potato_model"' for Potato dataset
  parser.add_argument('-path', '--folder', default = '"Potato_model"', type = str,
                     help='Path where the .json, .h5 files are saved')
  main(parser.parse_args())
  




  
      
    



