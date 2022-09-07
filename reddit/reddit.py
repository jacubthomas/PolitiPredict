from operator import truediv
from nltk.corpus import wordnet as wn
from cgitb import text
from nltk.stem import WordNetLemmatizer
from typing import List
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import gutenberg
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import os


class Reddit:
    def __init__(self, id: int, party: str, text: str):
        self.id = id
        self.party = party
        try:
            text = text.lower ()
            self.text = text
        except AttributeError:
            self.text = text
    def __str__(self):
        return f"{self.id}:{self.party}:{self.text}"

# Ingest dataset, creating objects from post, and appending to list by party
def processDataset ():
    all_posts = []
    dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
    dataframe1 = pd.read_excel(dir_path+'/reddit.xls')
    for row in dataframe1.iterrows():
        row = row[1]
        r = Reddit(row.Id, row.Political_Party, row.Title)
        all_posts.append (r)
    return all_posts
        
def splitByParty (all_posts):
    l_posts, r_posts = [], []
    for post in all_posts:
        if post.party == 'Liberal':
            l_posts.append (post)
        else:
            r_posts.append (post)
    return l_posts, r_posts

# Trim stop words from text
def stopWords (sentence: str):
    stop_words = set (stopwords.words ("english"))
    filtered_sentence = []
    try:
        words = word_tokenize (sentence)
        filtered_sentence = [w for w in words if not w in stop_words]
    except:
        pass
    return " ".join (filtered_sentence)


# seems to be a better approach to only include true hits
def find_the_features (document, word_features):
    # all unique words in a document
    words = set (document)
    features = {}
    for w in word_features:
        if w in words:
            features[w] = True

    return features


# UNUSED 
def allWordsInList (words):
    all_words = []
    for post in words:
        try:
            words = word_tokenize (post.text)
            for w in words:
                if w not in string.punctuation:
                    all_words.append (w)
        except:
            pass
    return all_words

# UNUSED 
def allWordsInPost (post):
    all_words = []
    try:
        words = word_tokenize (post.text)
        for w in words:
            if w not in string.punctuation:
                all_words.append (w)
    except:
        pass   
    return all_words