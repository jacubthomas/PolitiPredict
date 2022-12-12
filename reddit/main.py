# Ingest dataset, randomize/partition, create/train classifiers, pickle work
import trainNB

# Load pickled classifiers/data, test classifiers against test data, 
# update mysql with wrong predictions for run
import testNB

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_hub as hub

# import os
# import sys
# import pickle

# print(f"\n\n\n")


# model = tf.keras.models.load_model("16x16x1_bs256_nn.model")

# one_to_many_classifier_f = open(f"{dir_path}/pickled_algos/one_to_many_classifier_nb_classifier.pickle", "rb")
# one_to_many_classifier = pickle.load(one_to_many_classifier_f)
# one_to_many_classifier_f.close()

