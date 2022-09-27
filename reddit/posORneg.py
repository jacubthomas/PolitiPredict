# Enables us to import files outside this directory
import os
import sys
dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'nltk_tutorials'))
print (dir_path)
sys.path.insert(0, dir_path)
import random
import reddit
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# INCLUDE THIS TO TRAIN CLASSIFIERS
# import sentiment 

# INCLUDE THIS TO USE PRE-TRAINED CLASSIFIERS
import sentiment_mod as sm

# If you haven’t already, download the lexicon
# nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# how the dataset will be split up
training_partition = 0.7
testing_partition = 0.3

# good = "Obama was a wonderful president."
# bad = "Obama was a mediocre president."
# g_sent = sm.sentiment (good)
# b_sent = sm.sentiment (bad)
# print (f'Input: {good}, Result: {g_sent}\n')
# print (f'Input: {bad}, Result: {b_sent}\n')

# print (f"Enter a phrase or `q` to exit: ")
# for line in sys.stdin:
#     if 'q' == line.rstrip():
#         break
#     sent = sm.sentiment (line)
#     print (f'Input: {line}, Result: {sent}\n')
# print("Exit")



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

# print (f"Input: ({recombined_posts[0].text}, {recombined_posts[0].party})")
# print (f"Result: {sm.sentiment (recombined_posts[0].text)}")

# converts similar words to a root word
# lemmatizer = nltk.WordNetLemmatizer ()

# print (recombined_posts[0].text)
# recombined_posts[0].text = reddit.lemmatizeMe (lemmatizer, recombined_posts[0].text)

# print (recombined_posts[0].text)
# for post in recombined_posts:
#     post.text = reddit.lemmatizeMe (lemmatizer, post.text)

# whole int for partition by % specified at top of file
partition = int (len (recombined_posts) * training_partition)

# break into two sets by partition
training_set = recombined_posts[:partition]
testing_set = recombined_posts[partition:]

# Need a map for [(proper,party), value] initialized to 0
# training_map = {}
for t in training_set:
    try:
        t.vader = analyzer.polarity_scores (t.text)
        print (t.vader, t)
    except: 
        pass
    # print (t.vader, t.party)


    # tokenized = sent_tokenize (t.text)
    # tagged = ""
    
    # for s in tokenized:
    #     words = word_tokenize (s)
    #     tagged = nltk.pos_tag (words)
    #     # print (tagged)
    #     # proper_nouns = [item for item in tagged if item[1].startswith ('N')]
    #     # senti = sm.sentiment (s)
    #     vader = analyzer.polarity_scores(s)
    #     t
        
        # print (vader, proper_nouns, t.party)
        # print (senti, vader, proper_nouns, t.party)
        

# save_word_features = open(f"{dir_path}/pickled_algos/word_features5k.pickle","wb")
# pickle.dump(word_features, save_word_features)
# save_word_features.close()

# for sent in test[0]:
#     print (f"{sent}, {test[1]}\n")
# print (recombined_posts[0].text)
# documents = []
# for post in recombined_posts:
#     documents.append ( 
#         (
#             sent_tokenize (post.text), 
#             post.party
#         )
#     )
# for doc in documents:
#     for sentence in range (0, len (doc[0])):
#         doc[0][sentence] = nltk.pos_tagged(doc[0][sentence]) 


# print (documents[0])
# # Evaluate sentence by sentence for proper nouns and sentiment
# # Towards them. Update sum value in map along with proper noun
# for doc in documents:
#     tagged = nltk.pos_tag(doc[])
#     for t in tagged:
#         if t[1].startswith ('N'):
#             print (t[0])



# Run through test set. Attempt to classify sentence by sentence
# by evaluating proper nouns and looking at sentiment. Then
# compare classification with party actual and track (in)effectiveness.
