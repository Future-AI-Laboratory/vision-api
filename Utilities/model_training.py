'''
Containes the training steps involving model compile to defining metadata, hyperparameters, callbacls
Logging model training, callbacks using weights and biases
'''
import tensorflow as tf
import keras

# imports model build to access the model build and build log
import Potato_model_build as pmb
from tensorflow.keras.optimizer import Adam

class model_train:
  def __init__(self, learning_rate = 0.001, epochs = 10, batch_size = 32, loss_fn = 'sparse_categorical_crossentropy'):
    '''
    Initialize the model training hyperparameters
    '''
    # optimal LR
    self.learning_rate = 0.001 # initial LR here is 1e-8
    self.epochs = 10
    self.batch_size = 32
    self.loss_fn = 'sparse_categorical_crossentropy'
    self.optimizer = Adam(self.learning_rate)
    
    # define callback optimizer
    self.callback_config = {"log_weights": True,
                            "save_model": True,
                            "log_batch_frequency": 10,
                            "callback_config": callback_config

    }

    # define train configuration
    self.train_config = {"batch_size": self.batch_size,
                         "epochs": self.epochs,
                         "optimizer": self.optimizer  

    }

    print(f'Training configuration\n{self.train_config}')


  def train(self, training, validation, config):
    '''
    defining model training steps
    '''
    # model compiling
    model.compile(loss = self.loss_fn, optimizer = config.optimizer, metrics = ['accuracy']
    
    # defining model callbacks
    # reduce Learning Rate on Plateu
    # if it observes training curve(val_acc) is stuck in plateu, reduces LR by factor 0.2 with a patience = 3 
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_acc',
                                                    factor = 0.2,
                                                    patience =5
                                                    )

    # initially error is too high so starting from LR 1e-01 as the initial LR
    # learning rate scheduler to obtain optimum LR
    # schedule is exponential decaying LR

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-03/(epoch+1))



