'''
Author: Vision-team
Environment setup regarding GPU utilization check, 
'''
import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.python.client import device_lib
import logging


def environment_setup():
  try:
    tf.random.set_seed(999)
    np.random.seed(999)

    # improve reproducibility and make it more deterministic
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

    print('Reproducibility steps done!')

    # GPU utilization check
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    print(f'Device details\n{device_lib.list_local_devices()}')

  except Exception as e:
    logging.exception("message")


