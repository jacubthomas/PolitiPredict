from traceback import print_tb
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier	# wrapper for nltk to use SK-Learn
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

'''
Creating a voting system which collects classifications from various ML
algorithms to make a more reliable decision; Additionally, assign a
confidence level (%) with vote. If say 4/7 say 'Yes', 3/7 say 'No' 
confidence is low. if 7/7 say 'Yes', confidence is high.
'''

# '*' means allow for a variable number of args under classifiers
class VoteClassifier (ClassifierI):
	def __init__ (self, *classifiers):
		self._classifiers = classifiers

	def classify (self, features):
		votes = []

		for c in self._classifiers:
			v = c.classify (features)
			votes.append (v)
		
		return mode (votes)

	def confidence (self, features):
		votes = []

		for c in self._classifiers:
			v = c.classify (features)
			votes.append (v)
		
		choice_votes = votes.count (mode (votes))
		conf = (choice_votes / len (votes)) * 100
		return conf

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
print (f"LR_classifier Algo accuracy: {(nltk.classify.accuracy(LR_classifier, testing_set)) * 100}%")

SGD_classifier = SklearnClassifier (SGDClassifier ())
SGD_classifier.train (training_set)
print (f"SGD_classifier Algo accuracy: {(nltk.classify.accuracy(SGD_classifier, testing_set)) * 100}%")

SVC_classifier = SklearnClassifier (SVC ())
SVC_classifier.train (training_set)
print (f"SVC_classifier Algo accuracy: {(nltk.classify.accuracy(SVC_classifier, testing_set)) * 100}%")

LinearSVC_classifier = SklearnClassifier (LinearSVC ())
LinearSVC_classifier.train (training_set)
print (f"LinearSVC_classifier Algo accuracy: {(nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100}%")

NuSVC_classifier = SklearnClassifier (NuSVC ())
NuSVC_classifier.train (training_set)
print (f"NuSVC_classifier Algo accuracy: {(nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100}%")

# Composite classifier
voted_classifier = VoteClassifier (classifier, 
								   MNB_classifier,
								   BNB_classifier, 
								   LR_classifier,
								   SGD_classifier,
								   SVC_classifier, 
								   NuSVC_classifier)

print (f"voted_classifier accuracy: {(nltk.classify.accuracy(voted_classifier, testing_set)* 100)}%")
print (f"Classification: {voted_classifier.classify (testing_set[0][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[0][0])}%\n")

print (f"Classification: {voted_classifier.classify (testing_set[1][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[1][0])}%\n")

print (f"Classification: {voted_classifier.classify (testing_set[2][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[2][0])}%\n")

print (f"Classification: {voted_classifier.classify (testing_set[3][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[3][0])}%\n")

print (f"Classification: {voted_classifier.classify (testing_set[4][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[4][0])}%\n")

print (f"Classification: {voted_classifier.classify (testing_set[5][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[5][0])}%\n")