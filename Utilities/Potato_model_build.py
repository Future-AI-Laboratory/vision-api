'''
Author: Vision-team
Build the model using keras functional API and log it using Weights and Biases for versoning
'''
import logging
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

import keras
import wandb


# builds the CNN model
class model_build:
  # initialize the parameters to configure the model
  def __init__(self, input_shape = (256, 256, 3), num_classes = 3, artifact_id = 'Potato_model_artifacts'):
    # default input image shape = (256,256,3)
    self.input_shape = input_shape
    # deafult num_classes = 3
    self.num_classes = num_classes
    self.model_artifact = artifact_id

    # defining model config
    self.model_config = {"num_classes" : self.num_classes,
                    "input_shape" : self.input_shape  
    }



  def potato_model(self):
    '''
    The potato model arcgitecture is defined as a tested predefined structure
    There is no hyperparameter variables used in this configuration 
    '''
    try:
      inputs = keras.layers.Input(shape = self.input_shape)

      x = keras.layers.Conv2D(filters = 16, kernel_size=3, activation='relu', strides=1)(inputs)
      x = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(x)
      x = keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu', strides=1)(x)
      x = keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2)(x)
      x = keras.layers.Flatten()(x)
      x = keras.layers.Dense(128, activation='relu')(x)
      x = keras.layers.Dropout(rate = 0.3)(x)
      output = keras.layers.Dense(self.num_classes, activation='softmax')(x)

      model = keras.models.Model(inputs = inputs, outputs = output)

      print('Model Summary\n')
      model.summary()

      # save model structure as image
      model_file_name = 'Potato_CNN.jpg'
      print(f'Potato Model is saved in {model_file_name} file!')

      return model
      
    except Exception as e:
      logging.exception('message')
  
  # is called during training
  def build_model_and_log(self):
    '''
    Build the model and log it to wandb
    '''
    try:
      # initializing wandb artifact project with default name for CNN model
      with wandb.init(project = self.model_artifact, job_type = "initialize", config = self.model_config) as run:
        config = wandb.config

        model = self.potato_model()

        potato_artifact = wandb.Artifact(
          "potato_convnet", type = "model",
          description = "Potato_CNN_model",
          metadata = dict(config)
        )

        # this is a way to save the model 
        # model.save("initialized_potato_model.keras")
        # also we can save the model in wandb
        model.save("initialized_potato_model.keras")
        # add model file to model artifact
        potato_artifact.add_file("initialized_potato_model.keras")
        # save the initialized model to that file
        wandb.save("initialized_potato_model.keras")

      return self.model_config

    except Exception as e:
      logging.exception("messsage")

       








  

