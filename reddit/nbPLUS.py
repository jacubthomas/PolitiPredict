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

# how the dataset will be split up
training_partition = 0.7
testing_partition = 0.3

# retrieve all posts, as constructed reddit objects, lowercased texts
all_posts = reddit.processDataset ()

# partition posts by party
lib_posts, rep_posts = reddit.splitByParty (all_posts)

# shuffle lists, randomize data set each run to ensure results are consistent
random.shuffle(lib_posts)
random.shuffle(rep_posts)

# MAY REMOVE THIS LATER
# TRIM LIB POSTS LENGTH DOWN TO = REP POST LENGTH
lib_posts = lib_posts[:len (rep_posts)]

# grab number of posts from each party
len_all_posts = len (all_posts)
len_lib_posts = len (lib_posts)
len_rep_posts = len (rep_posts)

# recombine lists
recombined_posts = lib_posts + rep_posts

# reshuffle list to blend posts by party
random.shuffle (recombined_posts)

# remove stop words
for post in recombined_posts:
    post.text = reddit.stopWords (post.text)

# converts similar words to a root word
lemmatizer = nltk.WordNetLemmatizer ()
for post in recombined_posts:
    before = post.text
    post.text = reddit.lemmatizeMe (lemmatizer, post.text)
    after = post.text

''' 
    create a list of tuples composed of all words in a post, 
    and that party associated with the post
'''
documents = []
for posts in recombined_posts:
    documents.append ( 
        (
            reddit.allWordsInPost (posts), 
            posts.party
        )
    )
    

all_words = []
# gather all words for frequency distribution
for d in documents:
    for w in d[0]:
        all_words.append (w)

# map counts of word occurrences
all_words = nltk.FreqDist (all_words)

# list of top 3000 occurring words in posts
word_features = list (all_words.keys ())[:7500]

'''
    create a complex set of tuples, or rather features, by party.
    each feature consists of all the top words mapped to booleans
    indicating their presence in a given post.
    looks like: 
    (feature, party) => (map{ top_words, present_in_post}, party)
'''
featuresets = [(reddit.find_the_features (rev, word_features), category) for (rev,category)in documents]

random.shuffle (featuresets)

# whole int for partition by % specified at top of file
partition = int (len (featuresets) * training_partition)

# break into two sets by partition
training_set = featuresets[:partition]
testing_set = featuresets[partition:]

# train on training set
classifier = nltk.NaiveBayesClassifier.train (training_set)

# output accuracy
print (f"Naive Bayes Algo accuracy: {(nltk.classify.accuracy(classifier, testing_set)) * 100}%")

# capturing stdout from s_m_i_f to gather words w/ weights
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout

classifier.show_most_informative_features (5000)
output = new_stdout.getvalue()

weighted_features = {}
highest_values = reddit.updateWeightedFeatures (weighted_features, output)

# restore stdout
sys.stdout = old_stdout

print (f"NB: Most important features\n{highest_values}")

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
voted_classifier = VoteClassifier (classifier, 
								   MNB_classifier,
								   BNB_classifier, 
								   LR_classifier,
								   SGD_classifier,
								   SVC_classifier, 
								   NuSVC_classifier)

print (f"voted_classifier accuracy: {(nltk.classify.accuracy(voted_classifier, testing_set)* 100)}%")

print (f"{testing_set[0][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[0][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[0][0])}%\n")

print (f"{testing_set[1][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[1][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[1][0])}%\n")

print (f"{testing_set[2][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[2][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[2][0])}%\n")

print (f"{testing_set[3][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[3][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[3][0])}%\n")

print (f"{testing_set[4][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[4][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[4][0])}%\n")

print (f"{testing_set[5][1]}")
print (f"Classification: {voted_classifier.classify (testing_set[5][0])}, ")
print (f"Confidence: {voted_classifier.confidence (testing_set[5][0])}%\n")


print (f"Enter a phrase or `q` to exit: ")

for line in sys.stdin:
    if 'q' == line.rstrip():
        break
    line = reddit.processInput (line)
    tokenized = word_tokenize (line)
    og_nb_classification = reddit.handleClassification (weighted_features, line)
    print (f"statement is: {og_nb_classification}\n")
    # features = voted_classifier.featurize (tokenized, word_features)
    features = {}
    for word in tokenized:
        if word in word_features:
            features[word] = True
        else:
            features[word] = False
    print (f"Voted classification:{voted_classifier.classify(features)} -  Confidence: {voted_classifier.confidence(features)}\n")
    # print (f"statement is: {voted_classifier.classify (line)}\n")
print("Exit")

