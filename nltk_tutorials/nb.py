from traceback import print_tb
import nltk
import random
from nltk.corpus import movie_reviews

'''
# clever, but less readable
documents = [(list (movie_reviews.words (fileid)), category)
	for category in movie_reviews.categories ()
	for fileid in movie_reviews.fileids (category)]
'''
documents = []
for category in movie_reviews.categories ():
	for fileid in movie_reviews.fileids (category):
			documents.append ((list (movie_reviews.words (fileid)), category))



'''
# testing purposes
print (documents[1])
'''

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

print (f"Naive Bayes Algo accuracy: {(nltk.classify.accuracy(classifier, testing_set)) * 100}%")
classifier.show_most_informative_features (15)