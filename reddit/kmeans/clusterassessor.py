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

df = pd.read_excel('5DPCA_Clustered.xlsx')
for n in [2,5,10,50,100, 500]:
    print (n)
    newdf = df[['party' ,f'{n}-clusters']]
    clusters = []
    cluster_counts = {}
    correct, wrong = 0, 0
    for i in (range(0,n)):
        cluster_counts['Liberal'] = 0
        cluster_counts['Conservative'] = 0 
        for index, row in newdf.iterrows ():
            if row['party'] == 'Liberal':
                if row[f'{n}-clusters'] == i:
                    cluster_counts['Liberal'] = cluster_counts['Liberal'] + 1
            else:
                if row[f'{n}-clusters'] == i:
                    cluster_counts['Conservative'] = cluster_counts['Conservative'] + 1
        if cluster_counts["Liberal"] > cluster_counts["Conservative"]:
            clusters.append ('Liberal')
            # print (f'cluster {i} is {clusters[-1]} with {cluster_counts["Liberal"]},{cluster_counts["Conservative"]}')
            correct += cluster_counts["Liberal"]
            wrong += cluster_counts["Conservative"]
        else:
            clusters.append ('Conservative')
            # print (f'cluster {i} is {clusters[-1]} with {cluster_counts["Conservative"]}, {cluster_counts["Liberal"]}')
            correct += cluster_counts["Conservative"]
            wrong += cluster_counts["Conservative"]
    print (f'run {i}: correct={correct}, wrong={wrong}; accuracy={float(correct)/float(correct+wrong)}')
