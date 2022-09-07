
'''
# calculate data partitions by party 
partition_train_lib = int (training_partition * len_lib_posts)
partition_test_lib = len_lib_posts - partition_train_lib
partition_train_rep = int (training_partition * len_rep_posts)
partition_test_rep = len_rep_posts - partition_train_rep

# print findings thus far
print (f"total posts: {len_all_posts}; total libs: {len_lib_posts}; total reps: {len_rep_posts}\n")
print (f"partition libs - train: {partition_train_lib} test: {partition_test_lib}\n")
print (f"partition libs - train: {partition_train_rep} test: {partition_test_rep}\n")

# remove all stop words from every post
for post in lib_posts:
    post.text = reddit.stopWords(post.text)
for post in rep_posts:
    post.text = reddit.stopWords(post.text)


# partition list into sets
# train_lib_posts = lib_posts[:partition_train_lib]
# test_lib_posts = lib_posts[partition_train_lib:]
# train_rep_posts = rep_posts[:partition_train_rep]
# test_rep_posts = rep_posts[partition_train_rep:]

# gather all words in each set
all_words_lib = reddit.allWordsInList (lib_posts)
all_words_rep = reddit.allWordsInList (rep_posts)

# calculate frequency distribution of words in lists
# ... counts occurrences per word
lib_words_freq_dist = nltk.FreqDist (all_words_lib)
rep_words_freq_dist = nltk.FreqDist (all_words_rep)


lib_partition_top_words = int (len (lib_words_freq_dist) * .35)
rep_partition_top_words = int (len (rep_words_freq_dist) * .35)

# list of top partitioned words from each party
lib_top_words = list (lib_words_freq_dist.keys ())[:lib_partition_top_words]
rep_top_words = list (rep_words_freq_dist.keys ())[:rep_partition_top_words]

l_feature_set = [(reddit.find_features (post, lib_top_words), "Liberal") for post in lib_posts]
r_feature_set = [(reddit.find_features (post, rep_top_words), "Conservative") for post in rep_posts]

l_training_set = l_feature_set[:partition_train_lib]
r_training_set = r_feature_set[:partition_train_rep]
l_test_set = l_feature_set[partition_train_lib:]
r_test_set = r_feature_set[partition_train_rep:]

# print (l_training_set)

# l_training_set = l_feature_set[:500]
# r_training_set = r_feature_set[:500]
# l_test_set = l_feature_set[500:750]
# r_test_set = r_feature_set[500:750]

# all_training_set = []
# all_training_set.append (l_training_set)
# all_training_set.append (r_training_set)
all_training_set = l_training_set + r_training_set


# all_test_set = []
# all_test_set.append (l_test_set)
# all_test_set.append (r_test_set)
all_test_set = l_test_set + r_test_set

# print (len (l_training_set))
# print (len (r_training_set))
# print (len (all_training_set), " : ", all_training_set[0][1], " : ", all_training_set[1][1])
'''

'''
train_text = gutenberg.raw("bible-kjv.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# enables smart lemmatization using part of speech association
def lemmatizeMe(lemma: WordNetLemmatizer, tagged: tuple[str, str]) -> str:
    if tagged[1].startswith('J'):
        return lemma.lemmatize(tagged[0], wn.ADJ)
    elif tagged[1].startswith('V'):
        return lemma.lemmatize(tagged[0], wn.VERB)
    elif tagged[1].startswith('N'):
        return lemma.lemmatize(tagged[0], wn.NOUN)
    elif tagged[1].startswith('R'):
        return lemma.lemmatize(tagged[0], wn.ADV)
    else:
        return lemma.lemmatize(tagged[0])


def process_content(tknized):
    try:
        lemmatizer = WordNetLemmatizer()
        for i in tknized:
            words = nltk.word_tokenize(i)
            # tag tokenized words to tuple along with identified part of speech 
            tagged = nltk.pos_tag(words)
            
                # There are times where this is good, others notsomuch.
                # For bayes, we don't want punctuation.
                # For sentiment/semantic sentence evaluation, we do.
                # # remove all punctuation tuples
                # # tagged = [t for t in tagged if t[0] not in string.punctuation]
            
            # print(tagged)
            # print()
            for j in range(0, len(tagged)):
                tagged[j] = (lemmatizeMe(lemmatizer, tagged[j]), tagged[j][1])
            print(tagged)
            print()
    except Exception as e:
        print(str(e))


for p in l_posts[0:5]:
    tokenized = custom_sent_tokenizer.tokenize(p.text)
    process_content(tokenized)
#     print()
'''