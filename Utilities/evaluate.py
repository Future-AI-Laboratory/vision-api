'''
Author: Vision-team
Here we will evaluate the trained model performance using test generator
'''
import logging

import wandb
import os
import keras
import numpy as np

# sklearn evaluation modules
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
import matplotlib.cm as cm

from matplotlib import pyplot as plt
import seaborn as sns


class evaluate:
  def __init__(self, test_dataset = None, artifact_id = 'Potato_model_artifacts_new', trained_artifact_name = "potato_trained_model", k = 10):
    self.artifact_id = artifact_id
    self.trained_artifact_name = trained_artifact_name
    # passing the test generator or tes_gen dataframe iterator
    self.test_gen = test_dataset
    # pass k value to visualize hardest k examples
    self.k = k
  
  def classification_analysis(self, labels, preds):
    try:
      print(f'Classification Report:\n{metrics.classification_report(labels, preds)}\n')
      print(f'\nClassification accuracy: {metrics.accuracy_score(labels, preds)*100: 0.2f}\n')
      fig_cm = plt.figure(figsize=(15,5))
      fig_cm.suptitle('Confusion Metrics for the Model')
      sns.heatmap(confusion_matrix(labels, preds), annot = True)
    
    except Exception as e:
      logging.exception("message")


  def get_hardest_k_examples(self, model):
    try:
      # obtain input, output pairs
      x_batch, y_batch = next(self.test_gen) 
      # class probabilities
      class_probs = model(x_batch)
      # predictions (argmax acoss columns, axis  = 1)
      pred = np.argmax(class_probs, axis = 1)

      losses = keras.losses.sparse_categorical_crossentropy(y_batch, class_probs)
      # sort the losses, and fetch the index(arg/inverse) of the losses 
      argsort_losses = np.argsort(losses)

      # compute highest losses(top k) using the highest loss index
      highest_k_losses = np.array(losses)[argsort_losses[-self.k:]]
      hardest_k_examples = x_batch[argsort_losses[-self.k:]]
      true_labels = y_batch[argsort_losses[-self.k:]]

      self.classification_analysis(y_batch, pred)

      return highest_k_losses, hardest_k_examples, true_labels, pred
    
    except Exception as e:
      logging.exception("message")


  def evaluation(self, model):
    '''
    Evaluation
    '''
    try:
      # passing the test_gen argument here, so that it can be used seperately
      loss, accuracy = model.evaluate(self.test_gen)
      highest_losses, hardest_examples, true_labels, predictions = self.get_hardest_k_examples(model)

      return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions
    except Exception as e:
      logging.exception("message")


  def evaluate_and_log(self, config = None):
    '''
    Evaluates and logs the test results to wandb
    '''
    try:
      with wandb.init(project = self.artifact_id, job_type = "report", config = config) as run:
        
        # use the trained model artifact
        model_artifact = run.use_artifact(self.trained_artifact_name+":latest")
        # download the trained_model artifact
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, self.trained_artifact_name+".keras")
        # load the models using keras
        model = keras.models.load_model(model_path)
        
        # obtain all the evaluation parameters
        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = self.evaluation(model)

        run.summary.update({"loss": loss, "accuracy": accuracy})
        
        # log hardest examples using wandb
        wandb.log({"high-loss-example":
                  [wandb.Image(hard_example, caption = "Pred: "+str(pred) + ",Label: " + str(label))
                  for hard_example, pred, label in zip(hardest_examples, preds, true_labels)]})
    
    except Exception as e:
      logging.exception("message")

