import numpy as np
import tensorflow as tf
import random

def global_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

