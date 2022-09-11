from traceback import print_tb
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier	# wrapper for nltk to use SK-Learn
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
'''
Module of ML algorithms! 
Amongst other things, provides a better NB than nltk
'''

documents = []
for category in movie_reviews.categories ():
	for fileid in movie_reviews.fileids (category):
			documents.append ((list (movie_reviews.words (fileid)), category))

random.shuffle (documents)
all_words = []

# convert to lowercase
for w in movie_reviews.words ():
	all_words.append (w.lower ())

# map counts of word occurrences
all_words = nltk.FreqDist (all_words)

# list of top 3000 occurring words in movie_reviews
word_features = list (all_words.keys ())[:3000]

def find_features (document):
	# all unique words in a document
	words = set (document)
	features = {}

	# map all words to boolean, if present/absent from top 3000
	for w in word_features:
		features[w] = (w in words)
	
	return features

featuresets = [(find_features (rev), category) for (rev,category)in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]


print (len (training_set))
# posterior = prior occurences * liklihood / evidence

classifier = nltk.NaiveBayesClassifier.train (training_set)

# classifier_f = open ("naivebayes.pickle", "rb")
# classifier = pickle.load (classifier_f)
# classifier_f.close ()

print (f"Original Naive Bayes Algo accuracy: {(nltk.classify.accuracy(classifier, testing_set)) * 100}%")
classifier.show_most_informative_features (15)

# Sk-learn NB classifiers: MultinomialNB, GaussianNB, BernoulliNB

MNB_classifier = SklearnClassifier (MultinomialNB ())
MNB_classifier.train (training_set)
print (f"MNB_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(MNB_classifier, testing_set)) * 100}%")


# GNB_classifier = SklearnClassifier (GaussianNB ())
# GNB_classifier.train (training_set)
# print (f"GNB_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(GNB_classifier, testing_set)) * 100}%")

BNB_classifier = SklearnClassifier (BernoulliNB ())
BNB_classifier.train (training_set)
print (f"BNB_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(BNB_classifier, testing_set)) * 100}%")


# save_classifier = open ("naivebayes.pickle", "wb")
# pickle.dump (classifier, save_classifier)
# save_classifier.close ()

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC


LR_classifier = SklearnClassifier (LogisticRegression ())
LR_classifier.train (training_set)
print (f"LR_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(LR_classifier, testing_set)) * 100}%")

SGD_classifier = SklearnClassifier (SGDClassifier ())
SGD_classifier.train (training_set)
print (f"SGD_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(SGD_classifier, testing_set)) * 100}%")

SVC_classifier = SklearnClassifier (SVC ())
SVC_classifier.train (training_set)
print (f"SVC_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(SVC_classifier, testing_set)) * 100}%")

LinearSVC_classifier = SklearnClassifier (LinearSVC ())
LinearSVC_classifier.train (training_set)
print (f"LinearSVC_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100}%")

NuSVC_classifier = SklearnClassifier (NuSVC ())
NuSVC_classifier.train (training_set)
print (f"NuSVC_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100}%")
