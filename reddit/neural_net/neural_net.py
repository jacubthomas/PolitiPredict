import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

import os
import sys
import pickle

from tensorflow import keras
# from keras.wrappers.scikitlearn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

print(f"\n\n\n")

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import reddit

dir_path = os.path.abspath (os.path.dirname ( __file__ ))

# how the dataset will be split up
training_partition = 0.7
testing_partition = 0.3

recombined_posts_f = open(f"{dir_path}/../pickled_algos/recombined_posts_news.pickle", "rb")
recombined_posts = pickle.load(recombined_posts_f)
recombined_posts_f.close()

X = []
y = []

for _ in recombined_posts:
    X.append (_.text)
    y.append (_.party)

data =  {
    "posts": X,
    "party": y
}
df = pd.DataFrame(data)

# set_length = len (X)
                            
# print (df.head())

# plt.hist(df.party, bins=2)
# plt.title("Party histogram")
# plt.ylabel("N")
# plt.xlabel("Party")
# plt.show()

df["label"] = (df.party == "Conservative").astype(int)
df = df[['posts', 'label']]

train, val, test = np.split(df.sample(frac=1), [int(0.7*len(df)), int(0.85*len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=16):
    df = dataframe.copy()
    labels = df.pop("label")
    df = df['posts']
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train, False)
valid_data = df_to_dataset(val, False)
test_data = df_to_dataset(test)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

# print (hub_layer(list(train_data)[0][0]))

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(train_data)

model.evaluate(valid_data)

history = model.fit(train_data, epochs=10, validation_data=valid_data)

model.evaluate(test_data)

model.save('16x16x1_nn_news.model')

''' MODEL 2: LSTM'''

# encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
# encoder.adapt(train_data.map(lambda text, label: text))

# vocab = np.array(encoder.get_vocabulary())

# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         mask_zero=True
#     ),
#     tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(16, activation='relu', return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(4, activation='relu', return_sequences=False),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])

# model.evaluate(train_data)
# model.evaluate(valid_data)

# history = model.fit(train_data, epochs=10, validation_data=valid_data)

# model.evaluate(test_data)
# model.save('256x256x1_LSTM.model')

# def create_model (layers, activation):
#     model = tf.keras.Sequential()
#     embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
#     hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
#     model.add (hub_layer)
#     for i, nodes in enumerate (layers):
#         if i==0:
#             model.add (Dense (nodes, input_dim=train.shape[1]))
#             model.add (Activation (activation))
#         else:
#             model.add (Dense (nodes))
#             model.add (Activation (activation))
#     model.add (Dense (1))

#     model.compile ( optimizer="adam",
#                     loss="binary_crossentropy",
#                     metrics=['accuracy'])
#     return model

# model = KerasClassifier (build_fn=create_mode, verbose=0)

# layers = [[16], [16,16], [32,16,8,4,2], [64,16], [256,256], [1024,256,64,16,4] [1024,1024]]
# activations = ['sigmoid', 'relu']
# param_grid = dict( layers=layers,
#                    activation=activations,
#                    batch_size=[256,1024,4096],
#                    epochs=[10],
#                    learning_rate=1e-3)
# grid = GridSearchCV (estimator=model, param_grid=param_grid)

# grid_result = grid.fit (train.x, train.y)