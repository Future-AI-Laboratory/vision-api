'''
It analyzes the content of the dataset of particular crop-type
(This code block is optional, while running the ML pipeline)

Note: 1. This comparison does not cover or include the original vs augmented comparison
2. We dynamically store images of all the classes of particular type to a seperate image folder
3. We are not using image_data_generator.flow_from_directory(), as we need to do startified split
4. Store the image names to a dataframe, and image name should be a dynamic name while copying from source dataset folder
5. Issue with the dynamic name is it copies to a destination folder to a seperate name(Ex. source(118,119,120,121) to dst(118, 119, 12, 121)
6. To avoid that count the total images of that croptype present in the dataset for all classes
7. Count number of digits(ex. total 3000 images, digit count = 4)
8. Then, use integer precision to ensure the destination name as (0,1,2,3...) as (0000, 0001, 0002....,0100, ....2999)
9. To do integer precision use dynamic string (str = '{:'+str(digit_count)+'d}') (Ex: '{:4d}.jpg'.format(sample_count))
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array
import seaborn as sns
import shutil

# defining the data inspection class
class data_inspection:
  def __init__(self, dataset_folder, image_folder_path = 'images', classifiers_list = None, crop_type = None):
    '''
    For simplicity of flow_from_dataframe function
    After fetching each image info, the image is stored to a single folder
    'image_folder_path' defines the signle path
    '''
    # the dataset path (here master path of plant_village_dataset)
    self.dataset_path = dataset_folder
    # classifiers list(specifically classifiers folder)
    self.classifiers = classifiers_list
    # crop type
    self.crop_type = crop_type
    # destination image folder where images of particular crop type(for all classes) will be stored together
    # for the purpose of imagedatagenerator
    self.image_folder = image_folder_path
    
    if not os.path.isdir(self.image_folder):
      os.mkdir(self.image_folder)
      print(f'The main image folder "{self.image_folder}" is created for {self.crop_type}')

    else:
      print(f'"{self.image_folder}" is already present to copy the images as a whole!') 
    
    # calls sample_image_digit_count to return total 
    digit_count = self.sample_image_count()
    self.digit_count_str = '{:'+str(digit_count)+'d}.jpg'
    print(f'All the image names in the destination folder will be {digit_count} digit numbers!')
     

  def sample_image_count(self):
    '''
    Caounts total image samples of that crop_type correspondin to all the individual class
    '''
    # counts total image samples
    class_img_sample_count = 0
    for cls in self.classifiers:
      cls_path = os.path.join(self.dataset_path, cls)
      class_img_sample_count += len(os.listdir(cls_path))

    # length of total digits in the image samples count
    return len(str(class_img_sample_count))  

  def dataset_content(self):
    # checks the list of all disease classes available in the dataset/image folder
    disease_classes = os.listdir(self.dataset_path)
    print(f'Total Disease Classifiers {len(disease_classes)}\n')
    for i in disease_classes:
      print(f'{i}')
    
    print(f'\nThe available classifiers for {self.crop_type} crop is \n{self.classifiers}')
  
  def convert_img_to_array(self, image_path):
    '''
    Converts images to numpy array
    '''
    try:
      img = plt.imread(image_path)
      if img is not None:
        return img_to_array(img)
      else:
        return np.array([])
    
    except Exception as e:
      print(f'Error {e}')
      return None

  def fetch_img_info(self):
    '''
    Fetch the distribution of different classifiers and shapes
    While fetching individual image info, copy them to a single folder 'images'
    It will help to use imagedatagenerator.flow_from_dataframe()

    '''

    crop_classifiers_path = [os.path.join(self.dataset_path, disease) for disease in self.classifiers]
    print(f'{self.crop_type.upper()} directory names list {crop_classifiers_path}')
    
    # image array list
    #image_list = []
    # image labels list
    #label_list = []
    
    # label_dataframe
    self.label_df = pd.DataFrame(columns=self.classifiers)
    
    # image dataframe
    self.img_df = pd.DataFrame(columns=['Image', 'Shape', 'Label'])

    try:
      print('\n[INFO] Loading images ...')
      # total sample count
      sample_count = 0
      for sr, crop_disease_path in enumerate(crop_classifiers_path):
        print(f'[INFO] Processing {self.classifiers[sr]}')
        sample_list = os.listdir(crop_disease_path)
        #print(len(sample_list))
        for disease_sample in sample_list:
          # removing the .DS_store files from list, which contains the folder infos
          if disease_sample == '.DS_store':
            sample_list.remove(disease_sample)
        # categorical count
        count = 0
        for disease_sample in sample_list:
          disease_sample_path = os.path.join(crop_disease_path, disease_sample)
          if disease_sample_path.endswith(".jpg") == True or disease_sample_path.endswith(".JPG") == True:
            img = self.convert_img_to_array(disease_sample_path)
            # copy the image to 'images' folder (self.image_folder)
            # sample destination path of crop image(sample_dst_path_ci)
            #str_path = 
            sample_dst_path_ci = self.digit_count_str.format(sample_count)
            sample_dest_path = os.path.join(self.image_folder, sample_dst_path_ci)
            #print(sample_dest_path)
            shutil.copy(disease_sample_path, sample_dest_path)
            #image_list.append(img)
            #label_list.append(potato_classifiers[sr])
            self.img_df.loc[sample_count,'Image'], self.img_df.loc[sample_count,'Label'] = sample_dst_path_ci, self.classifiers[sr]
            self.img_df.loc[sample_count,'Shape'] = img.shape
            count += 1
            sample_count += 1

        self.label_df.loc[0, self.classifiers[sr]] = count
      
      self.label_shape_df = self.img_df[['Shape', 'Label']].value_counts()
      # checks whether there are multiple image shapes present or not
      # if shape_imbalance is true it returns, there is a shape imbalance in the dataset
      self.shape_imbalance = self.label_shape_df.shape[0] > len(self.classifiers)
      
      # image_list, label_list, 
      
      # checks whether all the images are copied to single image folder or not
      assert len(os.listdir(self.image_folder)) == sample_count
      print(f'All the {sample_count} images for {self.crop_type} are copied to image folder successfully, test passed!')
# 
      return self.label_df, self.img_df, self.shape_imbalance   
        
    except Exception as e:
      print(f'Error {e}')
      return None
  
  # target distribution visualize of dataset images
  # this is also optional, if visalize flag is enabled for this at the starting of the pipeline
  def distribution(self, counts_series, label = None):
    fig = plt.figure(figsize=(15,5))
    sns.barplot(x = counts_series.index, y = counts_series.values)
    plt.title(f'Target Distribution of the {self.crop_type} dataset', fontsize=14)
    plt.xlabel(f'{label} Distribution of different classes')
    plt.show()
  
  # this is also optional, if visalize flag is enabled for this at the starting of the pipeline
  # visualizes the distribution
  def distribution_vis(self):
    # returns pandas series with classes as keys and count as values for augmented dataset
    target_counts = self.img_df['Label'].value_counts()

    # shape analysis for augmented
    shape_counts = self.img_df['Shape'].value_counts()

    # visualize target distribution
    self.distribution(target_counts, label = 'Target')

    # visualize shapes distribution
    self.distribution(shape_counts, label = 'Shape')
    
    print(f'Image shape comparison between different categories \n\n{self.label_shape_df}\n')
    if self.shape_imbalance:
      print('There are imbalances in image shapes!')

      # plotting the multi-index dataframe's hist() plot, where label, and shape are two index
      self.label_shape_df.unstack(level = 1).plot(kind = 'bar', subplots = True, figsize = (10,10), 
                                            title = 'Visualization of image shape comparison between different categories')
      plt.show()

    else:
      print('All the images of the dataset are of same shape')



