'''
Author: Vision Team
Instruction: While editing this file, to show the effect of edit
must restart the colab environment
'''

import zipfile
import os

class invalid_link_exception(Exception):
  "Exception occurred when download link is invalid"
  
  def __init__(self, download_link, message = 'Dataset link is not valid'):
    self.link = download_link
    self.message = message
    super().__init__(self.message)


# Data Storage Class
class datastore():
  def __init__(self, download_link = None, download_flag = False):
    try:
      if download_flag:        
        command = 'wget '+ download_link
        os.system(command)
        
        if os.system(command)!=0:
          raise invalid_link_exception(download_link)

        else:
          print('{0} successfully downloaded...'.format(download_link))

      else:
        pass
      
    # exception occurred if fn attribute is missing
    except Exception as e:
      print('Exception occurred, check download link!')
    except OSError as e1:
      print(e1)
  

  def unzip(self, source, destination):
    '''
    Source: Zip file
    Destination: Destination of unzip folder
    '''
    try:

      self.destination = destination
      # create destination folder if not existed
      if not os.path.isdir(destination):
        os.mkdir(destination)

      # unzip
      with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(self.destination)
      
      print('Unzipped done successfully to folder "{0}"...'.format(self.destination))
      # returns dataset folder
      return self.destination

    except Exception as e2:
      print('Exception occurred while unzipping!')



