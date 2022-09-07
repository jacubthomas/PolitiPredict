import sys
import nltk
import reddit
import random

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
    post.text = reddit.stopWords(post.text)

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

# whole int for partition by % specified at top of file
partition = int (len (featuresets) * training_partition)

# break into two sets by partition
training_set = featuresets[:partition]
testing_set = featuresets[partition:]

# train on training set
classifier = nltk.NaiveBayesClassifier.train (training_set)

# output accuracy
print (f"Naive Bayes Algo accuracy: {(nltk.classify.accuracy(classifier, testing_set)) * 100}%")

# output top words
classifier.show_most_informative_features (100)

print (f"Enter a phrase or `q` to exit: ")

for line in sys.stdin:
    if 'q' == line.rstrip():
        break
    fs = reddit.find_the_features(line, word_features)
    label = classifier.prob_classify (fs)
    # print (f"probability: {label.prob (fs)}")
    print (f"statement is: {classifier.classify (fs)}\n")
    print (f"l prob dist: {label._prob_dict['Liberal']}; r prob dist: {label._prob_dict['Conservative']}\n")
print("Exit")

'''
These tests from random posts are being accurately predicted, however random text is less accurate

# federal grand jury indicts former trump adviser steve bannon contempt congress
# capitalism god way determining smart poor ron swanson
# donald trump jr. full speech cpac 2022 orlando
# how hospital scrubs reinforce sexist double standards
'''