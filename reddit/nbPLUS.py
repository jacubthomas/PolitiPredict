import io
import sys
import nltk
import pickle
import reddit
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier	# wrapper for nltk to use SK-Learn
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

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

def oneToManyVoter (phrase_length, classifier, training_set, testing_set):
    MNB_classifier = SklearnClassifier (MultinomialNB ())
    MNB_classifier.train (training_set)
    print (f"MNB_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(MNB_classifier, testing_set)) * 100}%")

    BNB_classifier = SklearnClassifier (BernoulliNB ())
    BNB_classifier.train (training_set)
    print (f"BNB_classifier Naive Bayes Algo accuracy: {(nltk.classify.accuracy(BNB_classifier, testing_set)) * 100}%")

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
    voted_classifier = VoteClassifier ( 
                                        # classifier, 
                                        MNB_classifier,
                                        BNB_classifier, 
                                        LR_classifier,
                                        SGD_classifier,
                                        SVC_classifier, 
                                        NuSVC_classifier)

    # print (f"{phrase_length} voted_classifier accuracy: {(nltk.classify.accuracy(voted_classifier, testing_set)* 100)}%")
    return voted_classifier

