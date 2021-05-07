'''
Author: Vision-team
This is used for data-preperation, train-test-split
Data preperation check
'''
import pandas as pd
from sklearn.model_selection import train_test_split

class data_preperation:
  def __init__(self, img_df, label_df, image_df_columns):
    # image_dataframe
    self.img_df = img_df
    # label_dataframe
    self.label_df = label_df
    self.columns = image_df_columns

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
      img_ids = self.img_df.loc[:, self.columns[0]]
      label_ids = self.img_df.loc[:, self.columns[1]]
      
      # cheks whether total image and 
      assert len(img_ids)==len(label_ids)
      print('No mismatch is there in the image and label list')
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

  def train_test_distribution_check(self, classifiers):
    assert list(self.df_train[self.columns[1]].value_counts.keys()) == list(self.df_train[self.columns[1]].value_counts.keys()) == classifiers
    print('Label check is done for train and test dataframe!')
    train_distribtion = list(self.df_train['Label'].value_counts().values)
    train_distribtion = list(self.df_train['Label'].value_counts().values)
    '''
    fig = plt.figure(figsize=(15,5))
    labels = ['Train distribution', 'Test distribution']
    fig.suptitle(f'Train_Test distribution for {label} dataset')
    for i in range(2):
      fig.add_subplot(1,2,i+1)
      g = sns.barplot(x=potato_classifiers, y = class_count[i]) 
      # display peak values on bar plot, create dataframe to do so
      #g.text(potato_classifiers[0], 800,1, color = 'black', ha = 'center')
      plt.xlabel(labels[i])
      
    
    # distribution study for original dataset
    train_test_distribution(y_train_original, y_test_original, label='Original')

    # distribution study for augmented dataset
    train_test_distribution(y_train_augmented, y_test_augmented, label='Augmented')
    '''