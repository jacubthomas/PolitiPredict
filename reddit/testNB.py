import os
import pickle
import reddit
import naivebayes

from nltk.tokenize import word_tokenize
# wrapper for nltk to use SK-Learn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

import mysql.connector

'''
    Overview: testNB loads pickled data and pre-trained classifiers to then
    brute-force test against a data set and outputs their performance. It also
    connects to a mysql db and inserts/updates posts the classifiers wrongly
    predict for assessment; posts that continue to be wrongly predicted may be
    noise or incorrectly labeled.
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

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="AdInfinitum2!"
)

mycursor = mydb.cursor()

mycursor.execute("USE Reddit")


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

one_to_many_classifier_f = open(f"{dir_path}/pickled_algos/one_to_many_classifier_nb_classifier.pickle", "rb")
one_to_many_classifier = pickle.load(one_to_many_classifier_f)
one_to_many_classifier_f.close()

two_to_many_classifier_f = open(f"{dir_path}/pickled_algos/two_to_many_classifier_nb_classifier.pickle", "rb")
two_to_many_classifier = pickle.load(two_to_many_classifier_f)
two_to_many_classifier_f.close()

three_to_many_classifier_f = open(f"{dir_path}/pickled_algos/three_to_many_classifier_nb_classifier.pickle", "rb")
three_to_many_classifier = pickle.load(three_to_many_classifier_f)
three_to_many_classifier_f.close()
'''
    End Load all pickled work
'''
# # Composite classifier
voted_classifier = VoteClassifier(
    one_word_nb_classifier, two_word_nb_classifier, three_word_nb_classifier)

length_testing = len (voted_classifier._classifiers[0].testing_set)


# # This is a brute-force method which runs through the testing set, one-by-one,
# # and makes a prediction based on the arg classifier; then outputs the results
# # + accuracy over the set. 
def assessMany (name, classifier): 
    correct, unsure, no_contest, wrong = 0, 0, 0, 0
    for i in range(0, length_testing):
        result = classifier.classifyCustom (classifier.testing_set[i][0])
        if result == "Moderate":
            unsure += 1
        elif result == "No Contest":
            no_contest += 1
        elif result == voted_classifier._classifiers[0].testing_set[i][1]:
            correct += 1
        else:
            wrong += 1
            post_id = findPostDetails (classifier.testing_set[i][0], classifier)
            exists_in_db = checkDBforID (post_id)
            updateDB (exists_in_db, post_id)

    
    naivebayes.outputResults (name ,correct, unsure, wrong, no_contest, length_testing)

# # This is a brute-force method which runs through the testing set, one-by-one,
# # and makes a prediction based on the arg classifier; then outputs the results
# # + accuracy over the set. 
def assessManytoMany (name, classifier): 
    correct, unsure, no_contest, wrong = 0, 0, 0, 0
    for i in range(0, length_testing):
        result = classifier.classify (voted_classifier._classifiers[0].testing_set[i][0])
        if result == "Moderate":
            unsure += 1
        elif result == "No Contest":
            no_contest += 1
        elif result == voted_classifier._classifiers[0].testing_set[i][1]:
            correct += 1
        else:
            wrong += 1
            post_id = findPostDetails (voted_classifier._classifiers[0].testing_set[i][0],
                                       voted_classifier._classifiers[0])
            exists_in_db = checkDBforID (post_id)
            updateDB (exists_in_db, post_id)

    
    naivebayes.outputResults (name ,correct, unsure, wrong, no_contest, length_testing)

def findPostDetails (post, classifier):
    for id,features in classifier.idToFeatureSet.items():
        if post == features[0]:
            return id


def checkDBforID (post_id):
    sql = f"SELECT * FROM WrongLists WHERE id = \'{post_id}\'" 
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    if len (myresult) == 1:
        return True
    return False

# Either insert a new wrongly predicted entry into the DB
# or update the count for an existing entry. This will be helpful
# for understanding which posts continue to challenge the
# classifier.
def updateDB (exists_in_db, post_id):

    reddit = findRedditByID (post_id)
    if exists_in_db == False:
        sql = "INSERT INTO WrongLists (id, posts, party, count) VALUES(%s, %s, %s, %s)"
        val = (post_id, reddit.text, reddit.party, 1)
        mycursor.execute(sql, val)
        mydb.commit()
    else:
        mycursor.execute (f"SELECT * FROM WrongLists WHERE id = \'{post_id}\'")
        myresult = mycursor.fetchone()
        sql = "UPDATE WrongLists SET count = %s WHERE id = %s"
        val = (int (myresult[3]) + 1, post_id)
        mycursor.execute(sql, val)
        mydb.commit()

def findRedditByID (post_id):
    for x in recombined_posts:
        if x.id == post_id:
            return x
    
    return None


assessMany ("one_word_nb_classifier", one_word_nb_classifier)
assessMany ("two_word_nb_classifier", two_word_nb_classifier)
assessMany ("three_word_nb_classifier", three_word_nb_classifier)

assessManytoMany ("one_to_many_classifier", one_to_many_classifier)
assessManytoMany ("two_to_many_classifier", two_to_many_classifier)
assessManytoMany ("three_to_many_classifier", three_to_many_classifier)

mycursor.close ()