import io
import sys
import nltk
import reddit
import random
from nltk.tokenize import word_tokenize

''' 
Yields a Naive-Bayes classifier, trained on and capable of evaluating n-length phrases
'''
class NBClassifier ():
    def __init__(self, posts, training_partition, phrase_length):
        self.posts = posts
        self.training_partition = training_partition
        self.phrase_length = phrase_length
        self.featureset = []
        self.idToFeatureSet = {}
        self.training_set = []
        self.testing_set = []
        self.weighted_features = {}
        self.classifier = trainClassifier (self)

    def classifyCustom (self, post):
            return self.classifier.classify (post)

''' 
    create a list of tuples composed of all words in a post, 
    and that party associated with the post
'''
def trainClassifier (self):
    documents = []
    for posts in self.posts:
        documents.append ( 
            (
                reddit.allWordPhrases (posts.text, self.phrase_length), 
                posts.party,
                posts.id
            )
        )

    all_words = []
    # gather all words for frequency distribution
    for d in documents:
        for w in d[0]:
            all_words.append (w)

    # map counts of word occurrences
    all_words = nltk.FreqDist (all_words)

    # sort by occurrences descending
    all_words = dict (
                        sorted (
                            all_words.items(),
                            key = lambda
                            item: item[1],
                            reverse = True
                        )
                    )

    # list of top 3000 occurring words in posts
    word_features = list (all_words.keys ())[:3000]

    '''
        create a complex set of tuples, or rather features, by party.
        each feature consists of all the top words mapped to booleans
        indicating their presence in a given post.
        looks like: 
        (feature, party) => (map{ top_words, present_in_post}, party)
    '''
    # self.featuresets = [(reddit.find_the_features (rev, word_features), category) for (rev,category)in documents]
    self.featuresets = []
    for (rev,category, id)in documents:
        temp = (reddit.find_the_features (rev, word_features), category)
        self.featuresets.append (temp)
        self.idToFeatureSet [id] = temp

    # whole int for partition by % specified at top of file
    partition = int (len (self.featuresets) * self.training_partition)

    # break into two sets by partition
    self.training_set = self.featuresets[:partition]
    self.testing_set = self.featuresets[partition:]

    # train on training set
    classifier = nltk.NaiveBayesClassifier.train (self.training_set)

    # output accuracy
    print (f"Naive Bayes {self.phrase_length}-word Algo accuracy: {(nltk.classify.accuracy(classifier, self.testing_set)) * 100}%")

    # capturing stdout from s_m_i_f to gather words w/ weights
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    classifier.show_most_informative_features (20000 * self.phrase_length)
    output = new_stdout.getvalue()

    highest_values = reddit.updateWeightedFeatures (self.weighted_features, output, self.phrase_length)

    # restore stdout
    sys.stdout = old_stdout

    print (f"{self.phrase_length}-word Most important features\n{highest_values}")
    return classifier

def classifyByWeightedFeatures (self, post):
    # Score by weights and output outcome
# def handleClassification2 (weighted_features: dict, line: str) -> str:
    l_score, r_score = 0,0

    for x, y in post.items ():        
        if x in self.weighted_features:
            feature = self.weighted_features[x]
            if feature[0].startswith("C"):
                r_score += float (feature[1])
            elif feature[0].startswith("L"):
                l_score += float (feature[1])
    
    if l_score == r_score == 0:
        return "No Contest"
    elif l_score > r_score:
        return "Liberal"
    elif l_score < r_score:
        return "Conservative"
    else: 
        return "Moderate"

'''
correct === accurate predictions
wrong === inaccurate predictions
unsure === Moderate === tie in votes
no contests entail there were no votes on either side
'''
def outputResults (classifier, correct, unsure, wrong, no_contest, length):
    print (f"[{classifier}] correct:{correct}/{length} || {(correct / length) * 100}")
    print (f"[{classifier}] unsure:{unsure}/{length} || {(unsure / length) * 100}")
    print (f"[{classifier}] wrong:{wrong}/{length} || {(wrong / length) * 100}")
    print (f"[{classifier}] no contest:{no_contest}/{length} || {(no_contest/length) * 100}\n\n")


# print (f"Enter a phrase or `q` to exit: ")

# for line in sys.stdin:
#     if 'q' == line.rstrip():
#         break
#     line = reddit.processInput (line)
#     tokenized = word_tokenize (line)
#     print (f"statement is: {reddit.handleClassification (weighted_features, line)}\n")
# print("Exit")
