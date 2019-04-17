import numpy as np
import tensorflow as tf  # pylint: ignore-module
import copy
import os

# methods that have been removed from tf_util

# ================================================================
# Global session	 # Global session
# ================================================================

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()


# ================================================================
# Save tensorflow summary
# ================================================================

def file_writer(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return tf.summary.FileWriter(dir_path, get_session().graph)


# ================================================================
# Saving variables
# ================================================================

def load_state(fname, var_list=None):
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(get_session(), fname)

def save_state(fname, var_list=None):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver(var_list=var_list)
    saver.save(get_session(), fname)