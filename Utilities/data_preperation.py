'''
Author: Vision-team
This is used for data-preperation, train-test-split
Data preperation check
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

# log errors/exceptios along with additional message
import logging

# logging module is used to log athe exception error with suggestion message

class data_preperation:
  def __init__(self, img_df, label_df, image_df_columns, classifiers):
    # image_dataframe
    self.img_df = img_df
    # label_dataframe
    self.label_df = label_df
    self.columns = image_df_columns
    self.classifiers = classifiers

    # image_df_columns contain list of column names in the img_df
    # image name column (column id = 0)
    # label column (column id = 1)
    #self.img_col, self.label_col = self.columns

  def stratified_split(self, split_ratio = [0.8,0.1,0.1]):
    '''
    Does the stratified train_test_split based on class labels
    Default test_size is taken aas 0.1 or 10% of the total dataset
    Default train: valid: test = 0.8:0.1:0.1
    '''
    try:
      # split ratio format [train:valid:test]
      self.split_ratio = split_ratio
      self.val_main_size = split_ratio[1] + split_ratio[2]
      self.test_size = split_ratio[2]/(split_ratio[1] + split_ratio[2]) 
      img_ids = self.img_df.loc[:, self.columns[0]]
      label_ids = self.img_df.loc[:, self.columns[1]]
      
      # cheks whether total image and 
      assert len(img_ids)==len(label_ids)
      print('No mismatch is there in the image and label list, check passed!')
      #print(img_ids, label_ids)
      
      # defining train dataframe, validation_dataframe, test_dataframe
      self.df_train = pd.DataFrame(columns = self.columns)
      # holds both validation and test set into it
      self.df_valid_main = pd.DataFrame(columns = self.columns)
      self.df_valid = pd.DataFrame(columns = self.columns)
      self.df_test = pd.DataFrame(columns = self.columns)

      # defining stratified train_valid_main split by shuffling the data
      # defining the random seed to avoid the issue due to reproducibility
      self.df_train[self.columns[0]], self.df_valid_main[self.columns[0]], self.df_train[self.columns[1]], self.df_valid_main[self.columns[1]] = train_test_split(img_ids, label_ids,
                                                                                                                             test_size = self.val_main_size,
                                                                                                                             random_state = 27,
                                                                                                                             stratify = label_ids,
                                                                                                                             shuffle = True          
      )
      
      
      # validation_main image names
      img_valid_main = self.df_valid_main[ self.columns[0]]
      # validation_main image labels
      label_valid_main = self.df_valid_main[self.columns[1]]
      print(img_valid_main, label_valid_main)

      # defining stratified valid_test split with shuffling the data(As we can not do startified split without shuffling)
      self.df_valid[self.columns[0]], self.df_test[self.columns[0]], self.df_valid[self.columns[1]], self.df_test[self.columns[1]] = train_test_split(img_valid_main, label_valid_main,
                                                                                                                             test_size = self.test_size,
                                                                                                                            random_state = 28,
                                                                                                                             stratify = label_valid_main,
                                                                                                                            shuffle = True # We can not do startified split without shuffling          
      ) 

      return self.df_train, self.df_valid, self.df_test

    except Exception as e:
      logging.exception("message")

  def train_test_distribution_check(self):
    try:  
      # checks whether lenghth of keys of 'Label' col of train and test matches with classifiers list 
      assert len(list(self.df_train[self.columns[1]].value_counts().keys())) == len(list(self.df_train[self.columns[1]].value_counts().keys())) == len(self.classifiers)
      for i in self.classifiers:
        # checks whether the individual class label is present in the list or not
        assert i in list(self.df_train[self.columns[1]].value_counts().keys()), list(self.df_test[self.columns[1]].value_counts().keys())
      print('Label check is done for train and test dataframe, check passed!\n')
      
      print(f'Train distribution\n{self.df_train[self.columns[1]].value_counts()}')
      print(f'\nValidation distribution\n{self.df_valid[self.columns[1]].value_counts()}')
      print(f'\nTest distribution\n{self.df_test[self.columns[1]].value_counts()}')
      
      # checks stratified split(the ratio of train and test samples count for each class/10 == (1-test_ratio))
      for cls in self.classifiers:
        assert abs(self.df_train[self.columns[1]].value_counts()[cls] / self.df_test[self.columns[1]].value_counts()[cls]) == (self.split_ratio[0]/self.split_ratio[2])
        assert abs(self.df_train[self.columns[1]].value_counts()[cls] / self.df_valid[self.columns[1]].value_counts()[cls]) == (self.split_ratio[0]/self.split_ratio[1]) 
      
      print('\nStratified train-test split check is passed!\n')
      
      train_distribtion = list(self.df_train[self.columns[1]].value_counts().values)
      valid_distribtion = list(self.df_valid[self.columns[1]].value_counts().values)
      test_distribtion = list(self.df_test[self.columns[1]].value_counts().values)
      
      # Train, test distribution validation
      fig = plt.figure(figsize=(25,5))
      labels = ['Train distribution', 'Validation Distribution', 'Test distribution']
      
      fig.suptitle(f'Train_Validation_Test distribution for the stratified split', fontsize = 18)
      fig.tight_layout()
      fig.add_subplot(131)
      g = sns.barplot(x = list(self.df_train[self.columns[1]].value_counts().keys()), y = train_distribtion) 
      # display peak values on bar plot, create dataframe to do so
      #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
      plt.xlabel(labels[0])

      fig.add_subplot(132)
      g = sns.barplot(x = list(self.df_valid[self.columns[1]].value_counts().keys()), y = valid_distribtion) 
      # display peak values on bar plot, create dataframe to do so
      #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
      plt.xlabel(labels[1])
      

      fig.add_subplot(133)
      g = sns.barplot(x = list(self.df_test[self.columns[1]].value_counts().keys()), y = test_distribtion) 
      # display peak values on bar plot, create dataframe to do so
      #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
      plt.xlabel(labels[2])

    except Exception as e:
      logging.exception('message')
 
  # horizontal, vertical flip is boolean(it's not Nne, either True or False)
  def data_generator(self, img_folder = None, rescale_param = 1./255, dtype = np.float32, batch_size = 64, interpolation = 'bicubic', classes = None, rotation_range = None, width_shift_range = None, height_shift_range = None, zoom_range = 0.0, horizontal_flip = False, vertical_flip = False, class_mode = 'categorical', target_size = (256,256), brightness_range = None):
    '''
    Creates train, validation, test data generator using keras ImageDataGenerator
    Note: The order of datagen class indices are reproducible with seed = 2020
    Multiple datageneration process will result same order of classes indices
    The class_indices will be edited to config.yaml for inference
    '''

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
          ) # brightness

      # train datagen
      self.train_gen = datagen.flow_from_dataframe(dataframe = self.df_train, 
                                                directory = img_folder, # specifies destination image folder where all the images are stored together
                                                x_col=self.columns[0], # image name column ('Image')
                                                y_col=self.columns[1], # image label column ('Label) 
                                                class_mode = self.class_mode, # taken as categorical 
                                                target_size = self.target_size, # default target size (256,256)
                                                batch_size= self.batch_size, # default batch size taken as 64
                                                seed=2020, # ensures better reproducibility
                                                classes = self.classifiers, # specifies class list in a pre-defined order
                                                interpolation = self.interpolation
                                                ) 
      
      # Validation datagen
      self.valid_gen = datagen.flow_from_dataframe(dataframe = self.df_valid, 
                                                directory = img_folder, # specifies destination image folder where all the images are stored together
                                                x_col=self.columns[0], # image name column ('Image')
                                                y_col=self.columns[1], # image label column ('Label) 
                                                class_mode = self.class_mode, # taken as categorical 
                                                target_size = self.target_size, # default target size (256,256)
                                                batch_size= self.batch_size, # default batch size taken as 64
                                                seed=2020, # ensures better reproducibility
                                                classes = self.classifiers, # specifies class list in a pre-defined order
                                                interpolation = self.interpolation # interpolation technique
                                                ) 
      
      # test datagen
      self.test_gen = datagen.flow_from_dataframe(dataframe = self.df_test, 
                                                directory = img_folder, # specifies destination image folder where all the images are stored together
                                                x_col=self.columns[0], # image name column ('Image')
                                                y_col=self.columns[1], # image label column ('Label) 
                                                class_mode = self.class_mode, # taken as categorical 
                                                target_size = self.target_size, # default target size (256,256)
                                                batch_size= self.batch_size, # default batch size taken as 64
                                                seed = 2020, # ensures better reproducibility
                                                classes = self.classifiers, # specifies class list in a pre-defined order
                                                interpolation = self.interpolation
                                                ) 
      
      return self.train_gen, self.valid_gen, self.test_gen

    except Exception as e:
      logging.exception("message")
  
  
  def datagen_check(self):
    '''
    This is used to check whether the data generation is done successfully
    '''
    try:
      assert self.train_gen.class_indices == self.valid_gen.class_indices == self.test_gen.class_indices 
      # classifier details, classifiers order check
      for cls in self.classifiers:
        assert list(self.train_gen.class_indices.keys()).count(cls) == self.classifiers.count(cls)
      
      labels_dict = self.train_gen.class_indices
      print('\nClass check is done successfully!')
      print(f'\nThe final label dictionary for the disease classes are follows,\nas defined by the generator will be used for the inference!\n{self.classifiers}')

      # batch_size check
      assert self.train_gen.batch_size == self.valid_gen.batch_size == self.test_gen.batch_size == self.batch_size
      print(f'\nCheck passed, batch size {self.batch_size}')

      # data type check
      assert self.train_gen.dtype == self.valid_gen.dtype == self.test_gen.dtype == self.dtype
      print(f'\nCheck, passed data type {self.dtype}')

      # class mode check
      assert self.train_gen.class_mode == self.valid_gen.class_mode == self.test_gen.class_mode == self.class_mode
      print(f'\nCheck passed, class mode {self.class_mode}')

      # target size check
      assert self.target_size == self.train_gen.target_size == self.valid_gen.target_size == self.test_gen.target_size
      print(f'\nCheck passed, target size {self.target_size}')
      
      # this classifiers dictionary will help to make label.txt for tflit endpoint
      return labels_dict


    except Exception as e:
      logging.exception('message')