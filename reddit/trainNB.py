import os
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

'''
    Overview: trainNB ingests the dataset from file, preprocesses the information,
    partitions the dataset, creates multiple classifiers, trains these classifiers on
    the training set, and pickles work as it goes. Upon running, it clears old pickled
    files, randomizes the dataset, retrains and writes pickles as new. 
'''

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
            v = c.classifyCustom(features[0])
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
'''

# retrieve all posts, as constructed reddit objects, lowercased texts
# all_posts = reddit.processDataset ()
# all_posts = reddit.processDatasetMYSQL ()
all_posts = reddit.processDatasetXLSX ()


# partition posts by party
lib_posts, rep_posts = reddit.splitByParty(all_posts)

# shuffle lists, randomize data set each run to ensure results are consistent
random.shuffle(lib_posts)
random.shuffle(rep_posts)

# grab number of posts from each party
len_all_posts = len(all_posts)
len_lib_posts = len(lib_posts)
len_rep_posts = len(rep_posts)

# equalize sets
lib_posts = lib_posts[:len_rep_posts]

print (f'\nTotal posts being trained on: {len(rep_posts)+len(lib_posts)}\n')

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

# after processing, posts may be empty or too small to evaluate
# these posts are trimmed.
for post in recombined_posts:
    if len (post.text) < 8:
        recombined_posts.remove (post)


# Create pickle dir if not exists
pickle_dir = f"{dir_path}/pickled_algos/"
# Check whether the specified path exists or not
isExist = os.path.exists (pickle_dir)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs (pickle_dir)

# Delete all previous pickles!
for f in os.listdir(pickle_dir):
    try:
        os.remove(os.path.join(pickle_dir, f))
    except:
        pass

# pickle work as we go
saverecombinedposts = open(f"{pickle_dir}/recombined_posts_news.pickle","wb")
pickle.dump(recombined_posts, saverecombinedposts)
saverecombinedposts.close()

one_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 1)
save_one_word_nb_classifier = open(f"{pickle_dir}/one_word_nb_classifier_news.pickle","wb")
pickle.dump(one_word_nb_classifier, save_one_word_nb_classifier)
save_one_word_nb_classifier.close()

two_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 2)
save_two_word_nb_classifier = open(f"{pickle_dir}/two_word_nb_classifier_news.pickle","wb")
pickle.dump(two_word_nb_classifier, save_two_word_nb_classifier)
save_two_word_nb_classifier.close()

three_word_nb_classifier = naivebayes.NBClassifier(
    recombined_posts, training_partition, 3)
save_three_word_nb_classifier = open(f"{pickle_dir}/three_word_nb_classifier_news.pickle","wb")
pickle.dump(three_word_nb_classifier, save_three_word_nb_classifier)
save_three_word_nb_classifier.close()

# 1 word voter
save_one_to_many_classifier = open(f"{pickle_dir}/one_to_many_classifier_nb_classifier_news.pickle","wb")
one_to_many_classifier = nbPLUS.oneToManyVoter (1,
                                                one_word_nb_classifier, 
                                                one_word_nb_classifier.training_set,
                                                one_word_nb_classifier.testing_set)
pickle.dump(one_to_many_classifier, save_one_to_many_classifier)
save_one_to_many_classifier.close()

# # 2 word voter
save_two_to_many_classifier = open(f"{pickle_dir}/two_to_many_classifier_nb_classifier_news.pickle","wb")
two_to_many_classifier = nbPLUS.oneToManyVoter (2,
                                                two_word_nb_classifier, 
                                                two_word_nb_classifier.training_set,
                                                two_word_nb_classifier.testing_set)
pickle.dump(two_to_many_classifier, save_two_to_many_classifier)
save_two_to_many_classifier.close()

# # 3 word voter
save_three_to_many_classifier = open(f"{pickle_dir}/three_to_many_classifier_nb_classifier_news.pickle","wb")
three_to_many_classifier = nbPLUS.oneToManyVoter (3,
                                                  three_word_nb_classifier, 
                                                  three_word_nb_classifier.training_set,
                                                  three_word_nb_classifier.testing_set)
pickle.dump(three_to_many_classifier, save_three_to_many_classifier)
save_three_to_many_classifier.close()
