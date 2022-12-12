# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
import sys
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
# import texthero as hero

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np

import multiprocessing
cores = multiprocessing.cpu_count()

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


print (f"cores : {cores}\n")

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

# training_partition = 0.7
# testing_partition = 0.3

recombined_posts_f = open(f"{dir_path}/../pickled_algos/recombined_posts.pickle", "rb")
recombined_posts = pickle.load(recombined_posts_f)
recombined_posts_f.close()

# df = pd.DataFrame ([o.__dict__ for o in recombined_posts])

# # print (df.head ())

# X = []
# y = []

# for _ in recombined_posts:
#     X.append (_.text)
#     y.append (_.party)
df = pd.DataFrame(columns=['id','party','text', 'tokenized'])
sentences = []
for x in recombined_posts:
    words = word_tokenize (x.text)
    row = [x.id, x.party, x.text, words]
    df.loc[len(df)] = row


'''
train, test = train_test_split(df, test_size=0.3, random_state=42)
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.party]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.party]), axis=1)

print (train_tagged.values[5])

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(100):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


# def vec_for_learning(model, tagged_docs):
#     sents = tagged_docs.values
#     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=100)) for doc in sents])
#     return targets, regressors


# y_train, X_train = vec_for_learning(model_dbow, train_tagged)
# y_test, X_test = vec_for_learning(model_dbow, test_tagged)
# logreg = LogisticRegression(n_jobs=1, C=1e5)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# from sklearn.metrics import accuracy_score, f1_score
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
'''








# tagged_data = [TaggedDocument(words=d, tags=[i]) for i, d in enumerate(df['tokenized'])]
# ''' Train doc2vec model '''
# model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=cores, epochs = 100)

# # ## Print model vocabulary
# print (model.dv[2])

# #build vocab
# model.build_vocab(tagged_data)

# # train model
# model.train(tagged_data, total_examples=model.corpus_count
#             , epochs=model.epochs)

# # # Save trained doc2vec model
# model.save("test_doc2vec.model")

# Load saved doc2vec model
model = Doc2Vec.load("test_doc2vec.model")













''' Tensorflow embedding

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
# module_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
model = hub.load(module_url)

sentence_embeddings = model(X)


print (sentence_embeddings)
'''
vectors = []
for i in range (0, len (model.dv)):
    vectors.append (np.array(model.dv[i]))
df2 = pd.DataFrame(vectors)
print (df.head())
print ('\n\n\n\n-----------------------------\n')
print (df2.head())

# # df2 = pd.DataFrame(pd.Series(vectors))
newdf = pd.concat([df, df2], axis=1)
newdf.to_excel('D2V_BOW_20V_100E.xlsx', sheet_name='Doc2Vec')
# df2 = pd.read_excel('20D_Doc2vec.xlsx')


# pca = PCA(n_components=5)
# principalComponents = pca.fit_transform(df2.iloc[:,6:26])
# # print (df2.iloc[:,6:26])
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1',
#                           'principal component 2',
#                           'principal component 3',
#                           'principal component 4',
#                           'principal component 5'])
# finalDf = pd.concat([principalDf, df[['party']]], axis = 1)
# finalDf.to_excel('20To5_PCA.xlsx', sheet_name='20ToPCA')


# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(df2) 
# principalDf = pd.DataFrame(data = z
#              , columns = ['principal component 1', 'principal component 2'])
# finalDf = pd.concat([principalDf, df[['party']]], axis = 1)
# print (finalDf)


# # plot with squared coordinates 
# plt.style.use('_mpl-gallery')
# fig, ax = plt.subplots()
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")


# colors = ['red', 'blue']
# for index,row in finalDf.iterrows ():
#     x = float(row['principal component 1'])
#     y = float(row['principal component 2'])
#     color_index = 0
#     if row['party'] == 'Liberal':
#         color_index = 1 
#     ax.scatter(x, y, color=colors[color_index])
# plt.show()
