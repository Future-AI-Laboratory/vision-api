import logging

import tensorflow as tf
import keras


import model_build as mb
from tensorflow.keras.optimizers import Adam

# used to visualize live training
from livelossplot import PlotLossesKeras

import wandb
import os



class model_train:
  def __init__(self, learning_rate = 0.0001, epochs = 50, batch_size = 32, loss_fn = 'sparse_categorical_crossentropy',artifact_id = 'model_artifacts_new', artifact_name = 'model_convnet', model_initialized_filename = "initialized_model.keras", trained_artifact_name = "trained_model"):
    '''
    Initialize the model training hyperparameters
    '''
    # optimal LR
    self.learning_rate = 0.001 # initial LR here is 1e-8
    self.epochs = 10
    self.batch_size = 32
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.optimizer = Adam(lr = self.learning_rate)
    self.model_artifact_id = artifact_id
    # model artifact name(can be passed as an argument from model_build)
    self.model_artifact_name = artifact_name
    # model trained artifact name
    self.model_trained_artifact_name = trained_artifact_name
    # model initialized filename(can be passed from model_build)
    self.model_initialized_filename = model_initialized_filename
    self 
    

    # define callback optimizer
    self.callback_config = {"log_weights": True,
                            "save_model": True,
                            "log_batch_frequency": 10
    }

    # define train configuration
    self.train_config = {"batch_size": self.batch_size,
                         "epochs": self.epochs,
                         "optimizer": self.optimizer,
                         "callback_config": self.callback_config  
    }

    print(f'Training configuration\n{self.train_config}')


  def train(self, model, training,valid,config):
    '''
    defining model training steps
    '''
    try:      
      
      # model compiling
      model.compile(loss = self.loss_fn, optimizer = self.optimizer, metrics = ['accuracy'])
      
      # defining model callbacks
      # reduce Learning Rate on Plateu
      # if it observes training curve(val_loss) is stuck in plateu, reduces LR by factor 0.1 with a patience = 5 
      reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                      factor = 0.1,
                                                      patience =5,
                                                      min_lr=0.00001,
                                                      mode='auto'
                                                      )

      # initially error is too high so starting from LR 1e-01 as the initial LR
      # learning rate scheduler to obtain optimum LR
      # schedule is exponential decaying LR

      lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-03/(epoch+1))
      print('STarting training....')
      # defining model fit 
      # passing train datagenerator, validation data generator
      model.fit(training, validation_data=valid, 
                batch_size = batch_size, # batch size is mentioned while building data generator
                epochs = epochs,
                callbacks = [wandb.keras.WandbCallback(**config["callback_config"]),
                            PlotLossesKeras(),
                            reduce_lr,
                            lr_schedule,
                            
                ] 
      )

    except Exception as e:
      logging.exception('message')

  def train_and_log(self, train_gen = None, valid_gen = None):
    '''
    Defining training and log steps to wandb
    '''
    try:
      # add train config while initializing the train with block
      with wandb.init(project = self.model_artifact_id, job_type = "train", config = self.train_config) as run:
        config = wandb.config
        
        # use the latest model artifact used previously
        #Note: Always use that syntax never use space, wrong: ("potato_convnet: latest"), correct: ("potato_convnet:latest")  
        
        model_artifact = run.use_artifact(self.model_artifact_name+":latest")
        print(f'Using the artifact - {self.model_artifact_name+":latest"}')
        # download latest version of the model artifact
        model_dir = model_artifact.download()
        # load the downloaded model(initialized) of model artifact from model_dir
        model_path = os.path.join(model_dir, self.model_initialized_filename)
        # load the model from model path using keras for training
        model = keras.models.load_model(model_path)
        
        # load model metadata to model_config
        model_config = model_artifact.metadata
        
        # update the config with that model config
        config.update(model_config)

        # start the training
        self.train(model, train_gen,valid_gen,config)

        # create new artifact model type for the trained model  
        model_artifact = wandb.Artifact(
          self.model_trained_artifact_name, type = "model",
          description = "CNN model trained with model.fit"
        )

        # save the trained model
        model_trained_filename = self.model_trained_artifact_name + ".keras"
        model.save(model_trained_filename)
        # add new file to the artifact with that name 
        model_artifact.add_file(model_trained_filename)
        # save to wandb
        wandb.save(os.path.join(wandb.run.dir, "model_trained_filename"))

        # log the artifact t train artifact
        run.log_artifact(model_artifact)

      return model, self.model_trained_artifact_name

    except Exception as e:
      logging.exception("message")