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

  def stratified_split(self, test_size = 0.1):
    '''
    Does the stratified train_test_split based on class labels
    Default test_size is taken aas 0.1 or 10% of the total dataset
    '''
    try:
      self.test_size = test_size
      img_ids = self.img_df.loc[:, self.columns[0]]
      label_ids = self.img_df.loc[:, self.columns[1]]
      
      # cheks whether total image and 
      assert len(img_ids)==len(label_ids)
      print('No mismatch is there in the image and label list, check passed!')
      #print(img_ids, label_ids)
      
      # defining train dataframe, test_dataframe
      self.df_train = pd.DataFrame(columns = self.columns)
      self.df_test = pd.DataFrame(columns = self.columns)

      # defining stratified train_test split
      self.df_train[self.columns[0]], self.df_test[self.columns[0]], self.df_train[self.columns[1]], self.df_test[self.columns[1]] = train_test_split(img_ids, label_ids,
                                                                                                                             test_size = test_size,
                                                                                                                             random_state = 27,
                                                                                                                             stratify = label_ids          
      ) 

      return self.df_train, self.df_test

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
    print(f'\nTest distribution\n{self.df_test[self.columns[1]].value_counts()}')
    
    # checks stratified split(the ratio of train and test samples count for each class/10 == (1-test_ratio))
    for cls in self.classifiers:
      assert abs(self.df_train[self.columns[1]].value_counts()[cls] / self.df_test[self.columns[1]].value_counts()[cls])/10 == (1 - self.test_size) 
    
    print('\nStratified train-test split check is passed!')
    
    train_distribtion = list(self.df_train[self.columns[1]].value_counts().values)
    test_distribtion = list(self.df_test[self.columns[1]].value_counts().values)
    
    # Train, test distribution validation
    fig = plt.figure(figsize=(15,5))
    labels = ['Train distribution', 'Test distribution']
    
    fig.suptitle(f'Train_Test distribution for the stratified split')
    fig.add_subplot(121)
    g = sns.barplot(x = list(self.df_train[self.columns[1]].value_counts().keys()), y = train_distribtion) 
    # display peak values on bar plot, create dataframe to do so
    #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
    plt.xlabel(labels[0])
      
    fig.add_subplot(122)
    g = sns.barplot(x = list(self.df_test[self.columns[1]].value_counts().keys()), y = test_distribtion) 
    # display peak values on bar plot, create dataframe to do so
    #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
    plt.xlabel(labels[1])
    