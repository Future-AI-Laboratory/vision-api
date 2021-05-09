'''
Author: Vision-team
This is used for data-preperation, train-test-split
Data preperation check
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

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
      print(e)

  def train_test_distribution_check(self):
    
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
    