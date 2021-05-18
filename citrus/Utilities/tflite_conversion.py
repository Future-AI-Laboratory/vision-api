'''
Author: Vision-team
Converting tflite model, label.txt
'''
import tensorflow as tf
import os
import logging

class model_conversion:
  def __init__(self, model_dir = "saved_model", tflite_dir="saved_tflite"):
    self.model_dir = model_dir
    self.tflite_dir = tflite_dir

    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    print(f"Model will be saved in {self.model_dir}")
    
    if not os.path.isdir(self.tflite_dir):
      os.mkdir(self.tflite_dir)
    print(f"Tflite version will be saved in {self.tflite_dir}")

  def tflite_conversion(self, model, labels_dict):
    try:
      tf.saved_model.save(model, self.model_dir)
      # define converted object
      converter = tf.lite.TFLiteConverter.from_saved_model(self.model_dir)
      tflite_model = converter.convert()
      # 'wb' = write binary
      tflite_path = os.path.join(self.tflite_dir, 'model.tflite')
      with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
      
      # label.txt file path
      label_path = os.path.join(self.tflite_dir, 'label.txt')
      labels_str = '\n'.join(sorted(labels_dict.keys()))

      with open(label_path, 'w') as f:
        f.write(labels_str)

      print(f'Tflite version is saved in {tflite_path}')
    except Exception as e:
      logging.exception("message")