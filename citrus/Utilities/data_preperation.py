import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

import logging

class data_preperation:

 
  # horizontal, vertical flip is boolean(it's not Nne, either True or False)
  def data_generator(self, img_directory = None,split=0.1, rescale_param = 1./255, dtype = np.float32, batch_size = 64, interpolation = 'bicubic', classes = None, rotation_range = None, width_shift_range = None, height_shift_range = None, zoom_range = 0.0, horizontal_flip = False, vertical_flip = False, class_mode = 'categorical', target_size = (220,220), brightness_range = None):

    try:
      # considering these two datagen parameters for checking the datagen  
      self.batch_size = batch_size
      self.dtype = dtype
      self.class_mode = class_mode
      self.target_size = target_size
      self.interpolation = interpolation

      # datagenerator object
      datagen = ImageDataGenerator(
          rotation_range = rotation_range, # rotation
          width_shift_range = width_shift_range, # horizontal shift
          height_shift_range = height_shift_range, # vertical shift
          zoom_range = zoom_range, # zoom
          rescale = rescale_param, # normalizing the pixels (rescale parameter is initialized to (1./255))
          horizontal_flip = horizontal_flip, # horizontal flip
          brightness_range = brightness_range,
          dtype = self.dtype,
          validation_split=split
          ) # brightness
      test_datagen = ImageDataGenerator(dtype = self.dtype,)

      self.train_gen = datagen.flow_from_directory(img_directory+"/Training", target_size=self.target_size, 
                                                   color_mode='rgb', classes=None,
                                                   class_mode = self.class_mode, batch_size= self.batch_size,
                                                   shuffle=True, seed=4, interpolation = self.interpolation,
                                                   subset="training"
                                                  )
      
      self.valid_gen = datagen.flow_from_directory(img_directory+"/Training", self.target_size, 
                                                   color_mode='rgb', classes=None,
                                                   class_mode = self.class_mode, batch_size= self.batch_size,
                                                   shuffle=True, seed=4, interpolation = self.interpolation,
                                                   subset="validation"
                                                  )

      self.test_gen = test_datagen.flow_from_directory(img_directory+"/Test", target_size=self.target_size, 
                                                   color_mode='rgb', classes=None,
                                                   class_mode = self.class_mode, batch_size= self.batch_size,
                                                   shuffle=True, seed=4, interpolation = self.interpolation,
                                                   subset="training"
                                                  )
      
      return self.train_gen,self.valid_gen,self.test_gen
      

    except Exception as e:
      logging.exception("message")

  
  def datagen_check(self):
    '''
    This is used to check whether the data generation is done successfully
    '''
    try:
      assert self.train_gen.class_indices == self.valid_gen.class_indices
      # classifier details, classifiers order check
      
      labels_dict = self.train_gen.class_indices
      print('\nClass check is done successfully!')
      # print(f'\nThe final label dictionary for the disease classes are follows,\nas defined by the generator will be used for the inference!\n{self.classifiers}')

      # batch_size check
      assert self.train_gen.batch_size == self.valid_gen.batch_size ==self.test_gen.batch_size== self.batch_size
      print(f'\nCheck passed, batch size {self.batch_size}')

      # data type check
      assert self.train_gen.dtype == self.valid_gen.dtype == self.test_gen.dtype==self.dtype
      print(f'\nCheck, passed data type {self.dtype}')


      # target size check
      assert self.target_size == self.train_gen.target_size == self.valid_gen.target_size == self.test_gen.target_size 
      print(f'\nCheck passed, target size {self.target_size}')
      
      # this classifiers dictionary will help to make label.txt for tflit endpoint
      return labels_dict


    except Exception as e:
      logging.exception('message')