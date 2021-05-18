import logging
import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras import datasets, layers, models


import keras
import wandb



class model_build:
  def __init__(self, input_shape = (220, 220, 3), num_classes = 3, artifact_id = 'model_artifacts_new', artifact_name = "model_convnet"):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.model_artifact_id = artifact_id
    self.model_artifact_name = artifact_name 
    self.model_config = {"num_classes" : self.num_classes,
                    "input_shape" : self.input_shape  
    }



  def model_arch(self):
    '''
    The model architecture is defined here
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
      output = keras.layers.Dense(self.num_classes)(x)

      model = keras.models.Model(inputs = inputs, outputs = output)

      print('Model Summary\n')
      model.summary()

      # save model structure as image
      model_file_name = 'Model_CNN.jpg'
      print(f'Model is saved in {model_file_name} file!')

      return model
      
    except Exception as e:
      logging.exception('Error during model building ')
  
  # is called during training
  def build_model_and_log(self):
    '''
    Build the model and log it to wandb
    '''
    try:
      # initializing wandb artifact project with default name for CNN model
      with wandb.init(project = self.model_artifact_id, job_type = "initialize", config = self.model_config) as run:
        config = wandb.config

        model = self.model_arch()

        model_artifact = wandb.Artifact(
          self.model_artifact_name, type = "model",
          description = "CNN_model",
          metadata = dict(config)
        )

        # this is a way to save the model 
        # model.save("initialized_model.keras")
        # also we can save the model in wandb
        model.save(os.path.join(wandb.run.dir, "initialized_model.keras"))
        # add model file to model artifact
        model_artifact.add_file("initialized_model.keras")
        # save the initialized model to that file
        wandb.save("initialized_model.keras")
        
        # log the artifact to wandb
        run.log_artifact(model_artifact)

      return self.model_config, self.model_artifact_id, self.model_artifact_name

    except Exception as e:
      logging.exception("Error during logging model to wandb")