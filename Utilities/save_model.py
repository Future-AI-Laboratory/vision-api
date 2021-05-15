'''
Authors: Vision-team
Save the model into json file(model architecture), model weights as .h5 file
'''
import keras
import os

class save_model:
  def __init__(self, model_dir = "Potato_model", model_json = "trained_model.json", model_weights = "trained_weights.h5"):
    self.model_dir = model_dir
    self.model_json = model_json
    self.model_weights = model_weights

    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    
    print(f'Saved model will be stored in "{self.model_dir}"') 

  def save_model(self, model):
    json_path = os.path.join(self.model_dir, self.model_json)
    weights_path = os.path.join(self.model_dir, self.model_weights)
    # saving json
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    print(f'Json file is saved to "{json_path}"')
    # saving weight file
    model.save_weights(weights_path)
    print(f'Model weights file is saved to "{weights_path}"')
    
