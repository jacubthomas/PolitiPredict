import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sklearn.cluster as cluster
# import pickle
# import texthero as hero
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import tensorflow as tf
# import tensorflow_hub as hub

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

dir_path = os.path.abspath (os.path.dirname ( __file__ ))

df = pd.read_excel('D2V_BOW_20V_100E.xlsx')

# print (df.iloc[:,0:5])
df2 = pd.DataFrame()

# print (df.iloc[:,5:25])

for n in [2,5,10,50,100]:
    kmeans = cluster.KMeans (n_clusters=n, init="k-means++")
    kmeans = kmeans.fit(df.iloc[:,5:25])
    df2[f'{n}-clusters'] = kmeans.labels_

df = pd.concat([df, df2], axis=1)
# print (df2)
# print (df2['3-clusters'].value_counts ())
df.to_excel('D2V_BOW_20V_100E.xlsx', sheet_name='Doc2Vec')