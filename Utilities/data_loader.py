import zipfile
import os

class invalid_link_exception(Exception):
  "Exception occurred when download link is invalid"
  
  def __init__(self, download_link, message = 'Dataset link is not valid'):
    self.link = download_link
    self.message = message
    super().__init__(self.message)



class dataloader():
  def __init__(self, download_link):
    try:
      command = 'wget '+ 'https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded'
      os.system(command)
      '''
      if os.system(command)!=0:
        print(1)
        raise invalid_link_exception(download_link)
      '''
    # exception occurred if fn attribute is missing
    except Exception as e:
      print('Exception occurred, check download link!')
    except OSError as e1:
      print(e1)

