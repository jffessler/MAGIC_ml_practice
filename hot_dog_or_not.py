import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_datasets as tfds
# from tensorflow.python.keras import datasets, layers, models
from tensorflow.python.keras import layers, models
# from tensorflow.python.keras import datasets


ds, ds_info = tfds.load('food101', shuffle_files=True, as_supervised=True, with_info=True)

train_ds, valid_ds = ds['train'], ds['validation']

fig = tfds.show_examples(train_ds,ds_info)