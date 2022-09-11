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

# enables smart lemmatization using part of speech association
def lemmatizeMe(lemma: WordNetLemmatizer, post: str) -> str:
    words = word_tokenize (post)
    tagged = nltk.pos_tag(words)
    for w in range (0, len (tagged)):
        if tagged[w][1].startswith('J'):
            words[w] = lemma.lemmatize(tagged[w][0], wn.ADJ)
        elif tagged[w][1].startswith('V'):
            words[w] = lemma.lemmatize(tagged[w][0], wn.VERB)
        elif tagged[w][1].startswith('N'):
            words[w] = lemma.lemmatize(tagged[w][0], wn.NOUN)
        elif tagged[w][1].startswith('R'):
            words[w] = lemma.lemmatize(tagged[w][0], wn.ADV)
        else:
            words[w] = lemma.lemmatize(tagged[w][0])
    return " ".join(words)

# seems to be a better approach to only include true hits
def find_the_features (document, word_features):
    # all unique words in a document
    words = set (document)
    features = {}
    for w in word_features:
        if w in words:
            features[w] = True

    return features

def processInput (line: str) -> str:
    try:
        line = line.lower ()
    except AttributeError:
        pass
    line = stopWords (line)
    lemmatizer = WordNetLemmatizer ()
    line = lemmatizeMe (lemmatizer, line)
    return line

# Score by weights and output outcome
def handleClassification (weighted_features: dict, line: str) -> str:
    l_score, r_score = 0,0
    line_as_list = line.split()
    for x in line_as_list:
        if weighted_features.get(x) != None:
            if weighted_features[x][0].startswith('L'):
                l_score += float (weighted_features[x][1])
            else:
                r_score += float (weighted_features[x][1])
    if l_score > r_score:
        return "Liberal"
    else:
        return "Conservative"

# duplicated code as above to include/test additional sk-algos
def classify (weighted_features: dict, line: str) -> str:
    l_score, r_score = 0,0
    line_as_list = line.split()
    for x in line_as_list:
        if weighted_features.get(x) != None:
            if weighted_features[x][0].startswith('L'):
                l_score += float (weighted_features[x][1])
            else:
                r_score += float (weighted_features[x][1])
    if l_score > r_score:
        return f"Liberal: {l_score} > {r_score}"
    else:
        return f"Conservative: {l_score} < {r_score}"

# populate map with key = word, value (party, weight)
# returns 25 key words with weights
def updateWeightedFeatures (weighted_features: dict, output: str) -> str:
    highest_values = ""
    output_as_list = output.split ()
    output_as_list = output_as_list[3:]
    for i in range (0, int (len (output_as_list) / 10)):
        weighted_features[output_as_list[i*10]] = (
            output_as_list[i*10+5],
            output_as_list[i*10+7]
        )
        if i < 25:
            highest_values += f"{output_as_list[i*10]} : {output_as_list[i*10+5]} : {output_as_list[i*10+7]}\n"

    return highest_values

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