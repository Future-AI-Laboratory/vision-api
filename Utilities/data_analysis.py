'''
It analyzes the content of the dataset of particular crop-type
(This code block is optional, while running the ML pipeline)
'''
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array


class data_inspection:
  def __init__(self, dataset_folder, classifiers_list = None, crop_type = None):
    self.dataset_path = dataset_folder
    self.classifiers = classifiers_list
    self.crop_type = crop_type
    
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
    '''

    crop_classifiers_path = [os.path.join(self.dataset_path, disease) for disease in self.classifiers]
    print(f'{self.crop_type} directory names list {crop_classifiers_path}')
    
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
            #image_list.append(img)
            #label_list.append(potato_classifiers[sr])
            self.img_df.loc[sample_count,'Image'], self.img_df.loc[sample_count,'Label'] = disease_sample, self.classifiers[sr]
            self.img_df.loc[sample_count,'Shape'] = img.shape
            count += 1
            sample_count += 1
        self.label_df.loc[0, self.classifiers[sr]] = count
      
      # image_list, label_list, 
      return self.label_df, self.img_df   
        
    except Exception as e:
      print(f'Error {e}')
      return None
