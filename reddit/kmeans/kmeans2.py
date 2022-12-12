import os
import sys
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

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

import reddit
# import mysql.connector
# recombined_posts = reddit.processDataset ()
# recombined_posts = reddit.processDatasetMYSQL ()

# print(len(db_posts))
# print(db_posts[:1][0].id)
# print(db_posts[:1][0].party)
# print(db_posts[:1][0].text)

# training_partition = 0.7
# testing_partition = 0.3

recombined_posts_f = open(f"{dir_path}/../pickled_algos/recombined_posts_news.pickle", "rb")
recombined_posts = pickle.load(recombined_posts_f)
recombined_posts_f.close()

# training_set = recombined_posts[:int(training_partition*len(recombined_posts))]

df = pd.DataFrame(columns=['id','party','text'])
print (df)
sentences = []
for x in recombined_posts:
    sentences.append(x.text)
    row = [x.id, x.party, x.text]
    df.loc[len(df)] = row

print (df)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)
words = vectorizer.get_feature_names()


'''Elbow method'''
# from sklearn.cluster import KMeans
# # wcss = []
# # for i in range(1,50):
# #     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
# #     kmeans.fit(X)
# #     wcss.append(kmeans.inertia_)
# # plt.plot(range(1,11),wcss)
# # plt.title('The Elbow Method')
# # plt.xlabel('Number of clusters')
# # plt.ylabel('WCSS')
# # plt.savefig('elbow.png')
# # plt.show()

# for i in range(2,)

numclusters = 10
kmeans = KMeans(n_clusters = numclusters, n_init = 20)
kmeans.fit(X)

df['cluster'] = kmeans.labels_

df.to_excel(f'kmeans2_{numclusters}_20iterations_news.xlsx')

numclusters = 25
kmeans = KMeans(n_clusters = numclusters, n_init = 20)
kmeans.fit(X)

df['cluster'] = kmeans.labels_

df.to_excel(f'kmeans2_{numclusters}_20iterations_news.xlsx')

numclusters = 50
kmeans = KMeans(n_clusters = numclusters, n_init = 20)
kmeans.fit(X)

df['cluster'] = kmeans.labels_

df.to_excel(f'kmeans2_{numclusters}_20iterations_news.xlsx')