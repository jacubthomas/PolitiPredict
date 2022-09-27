import io
import os
import sys
import nltk
import pickle
import reddit
import naivebayes
import nbPLUS
import random
from nltk.tokenize import word_tokenize
# wrapper for nltk to use SK-Learn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode



class VoteClassifier (ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self, i):
        votes = []
        for c in self._classifiers:
            features = c.testing_set[i]
            v = c.classify(features[0])
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = (choice_votes / len(votes)) * 100
        return mode(votes), conf

dir_path = os.path.abspath (os.path.dirname ( __file__ ))

# how the dataset will be split up
training_partition = 0.7
testing_partition = 0.3

'''
    Train and pickle work
    comment this running from last pickle algos

# retrieve all posts, as constructed reddit objects, lowercased texts
all_posts = reddit.processDataset()

# partition posts by party
lib_posts, rep_posts = reddit.splitByParty(all_posts)

# shuffle lists, randomize data set each run to ensure results are consistent
random.shuffle(lib_posts)
random.shuffle(rep_posts)

# grab number of posts from each party
len_all_posts = len(all_posts)
len_lib_posts = len(lib_posts)
len_rep_posts = len(rep_posts)

# recombine lists
recombined_posts = lib_posts + rep_posts

# reshuffle list to blend posts by party
random.shuffle(recombined_posts)

# remove stop words
for post in recombined_posts:
    post.text = reddit.stopWords(post.text)

# converts similar words to a root word
lemmatizer = nltk.WordNetLemmatizer()
for post in recombined_posts:
    before = post.text
    post.text = reddit.lemmatizeMe(lemmatizer, post.text)
    after = post.text

saverecombinedposts = open(f"{dir_path}/pickled_algos/recombined_posts.pickle","wb")
pickle.dump(recombined_posts, saverecombinedposts)
saverecombinedposts.close()

one_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 1)
save_one_word_nb_classifier = open(f"{dir_path}/pickled_algos/one_word_nb_classifier.pickle","wb")
pickle.dump(one_word_nb_classifier, save_one_word_nb_classifier)
save_one_word_nb_classifier.close()

two_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 2)
save_two_word_nb_classifier = open(f"{dir_path}/pickled_algos/two_word_nb_classifier.pickle","wb")
pickle.dump(two_word_nb_classifier, save_two_word_nb_classifier)
save_two_word_nb_classifier.close()

three_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 3)
save_three_word_nb_classifier = open(f"{dir_path}/pickled_algos/three_word_nb_classifier.pickle","wb")
pickle.dump(three_word_nb_classifier, save_three_word_nb_classifier)
save_three_word_nb_classifier.close()

    End train and pickle work
'''

'''
    Load all pickled work
    comment this out if retraining algos
'''
recombined_posts_f = open(f"{dir_path}/pickled_algos/recombined_posts.pickle", "rb")
recombined_posts = pickle.load(recombined_posts_f)
recombined_posts_f.close()
one_word_nb_classifier_f = open(f"{dir_path}/pickled_algos/one_word_nb_classifier.pickle", "rb")
one_word_nb_classifier = pickle.load(one_word_nb_classifier_f)
one_word_nb_classifier_f.close()
two_word_nb_classifier_f = open(f"{dir_path}/pickled_algos/two_word_nb_classifier.pickle", "rb")
two_word_nb_classifier = pickle.load(two_word_nb_classifier_f)
two_word_nb_classifier_f.close()
three_word_nb_classifier_f = open(f"{dir_path}/pickled_algos/three_word_nb_classifier.pickle", "rb")
three_word_nb_classifier = pickle.load(three_word_nb_classifier_f)
three_word_nb_classifier_f.close()
'''
    End Load all pickled work
'''
# Composite classifier
voted_classifier = VoteClassifier(
    one_word_nb_classifier, two_word_nb_classifier, three_word_nb_classifier)

length_testing = len (voted_classifier._classifiers[0].testing_set)
# for i in range(0, len(test_set))[:10]:
#     result = voted_classifier.confidence(i)
#     print(f"classification: {result[0]}  confidence: {result[1]} actual: {test_set[i].party}\n")

correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = voted_classifier.confidence(i)
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == voted_classifier._classifiers[0].testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("voted_classifier",correct, unsure, wrong, no_contest, length_testing)

# These are using self-written classifying algo
correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = naivebayes.classifyByWeightedFeatures (one_word_nb_classifier,
                                                    one_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == voted_classifier._classifiers[0].testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("CUSTOM one_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)

correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = naivebayes.classifyByWeightedFeatures (two_word_nb_classifier,
                                                    two_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == two_word_nb_classifier.testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("CUSTOM two_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)

correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = naivebayes.classifyByWeightedFeatures (three_word_nb_classifier,
                                                    three_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == voted_classifier._classifiers[0].testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("CUSTOM three_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)

# # idea: test each to see how accurate they are in predicting each party. Weight each classifiers vote
# # based on their accuracy for a given party. This should help with tie-breaking.

# # idea: if three votes, weight three highest as it is least likely.

# These use nltk classifier
correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = one_word_nb_classifier.classify (one_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == one_word_nb_classifier.testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("BUILT-IN one_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)

correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = two_word_nb_classifier.classify (two_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == voted_classifier._classifiers[0].testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("BUILT-IN two_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)

correct, unsure, no_contest, wrong = 0, 0, 0, 0
for i in range(0, length_testing):
    result = three_word_nb_classifier.classify (three_word_nb_classifier.testing_set[i][0])
    if result == "Moderate":
        unsure += 1
    elif result == "No Contest":
        no_contest += 1
    elif result == voted_classifier._classifiers[0].testing_set[i][1]:
        correct += 1
    else:
        wrong += 1

naivebayes.outputResults ("BUILT-IN three_word_nb_classifier", correct, unsure, wrong, no_contest, length_testing)


# 1 word voter
nbPLUS.oneToManyVoter (1,
                       one_word_nb_classifier, 
                       one_word_nb_classifier.training_set,
                       one_word_nb_classifier.testing_set)
# 2 word voter
nbPLUS.oneToManyVoter (2,
                       two_word_nb_classifier, 
                       two_word_nb_classifier.training_set,
                       two_word_nb_classifier.testing_set)
# 3 word voter
nbPLUS.oneToManyVoter (3,
                       two_word_nb_classifier, 
                       two_word_nb_classifier.training_set,
                       two_word_nb_classifier.testing_set)

'''
End of comment out for training.
'''