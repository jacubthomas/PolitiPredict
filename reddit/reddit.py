from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import os
import mysql.connector

class Reddit:
    def __init__(self, id: int, party: str, text: str, comments = None):
        self.id = id
        self.party = party
        try:
            string_encode = text.encode("ascii", "ignore")
            self.text = string_encode.decode()
            self.text = self.text.lower ()
            self.text = self.text.translate(
                str.maketrans('', '', string.punctuation)
                )
            if pd.notnull (comments):
                try:
                    string_encode = comments.encode("ascii", "ignore")
                    self.comments = string_encode.decode ()
                    self.comments = self.comments.lower ()
                    self.comments = self.comments.translate(
                        str.maketrans('', '', string.punctuation)
                        )
                    self.text += self.comments
                except:
                    print (f'error: id is {id}')
        except AttributeError:
            if isinstance(text, int):
                self.text = ' '
            else:
                self.text = text
            if pd.notnull (comments):
                try:
                    string_encode = comments.encode("ascii", "ignore")
                    self.comments = string_encode.decode ()
                    self.comments = self.comments.lower ()
                    self.comments = self.comments.translate(
                        str.maketrans('', '', string.punctuation)
                        )
                    self.text += self.comments
                except:
                    print (f'error: id is {id}')
    def __str__(self):
        return f"{self.id}:{self.party}:{self.text}"

# Ingest dataset, creating objects from post, and appending to list by party
def processDataset ():
    all_posts = []
    dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
    dataframe1 = pd.read_excel(dir_path+'/reddit.xls')
    for row in dataframe1.iterrows():
        row = row[1]
        r = Reddit(row.Id, row.Political_Party, row.Title, row.Text)
        all_posts.append (r)
    return all_posts

# Ingest dataset, creating objects from post, and appending to list by party
def processDatasetMYSQL ():
    all_posts = []
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="AdInfinitum2!"
    )
    mycursor = mydb.cursor()
    mycursor.execute("USE Reddit")
    sql = "SELECT * FROM Posts"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    for result in myresult:
        result = Reddit(result[0], 
                        result[2],
                        result[1],
                        result[4]
                        )
        all_posts.append (result)

    # close connection to DB
    mycursor.close ()
    return all_posts

# Ingest dataset, creating objects from post, and appending to list by party
def processDatasetXLSX ():
    all_posts = []
    index = 0
    dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
    dataframe1 = pd.read_excel(dir_path+'/News.xlsx')
    for row in dataframe1.iterrows():
        row = row[1]
        r = Reddit(index, row.party, row.text)
        # print (f'{index}: {row.party}, {row.text}\n')
        all_posts.append (r)
        index += 1
    return all_posts

def splitByParty (all_posts):
    l_posts, r_posts = [], []
    for post in all_posts:
        if post.party == 'Liberal':
            l_posts.append (post)
        else:
            r_posts.append (post)
    return l_posts, r_posts

# Trim stop words from text
def stopWords (sentence: str):
    stop_words = set (stopwords.words ("english"))
    filtered_sentence = []
    try:
        words = word_tokenize (sentence)
        filtered_sentence = [w for w in words if not w in stop_words]
    except:
        pass
    return " ".join (filtered_sentence)

# enables smart lemmatization using part of speech association
def lemmatizeMe(lemma: WordNetLemmatizer, post: str) -> str:
    try:
        words = word_tokenize (post)
        tagged = nltk.pos_tag(words)
        for w in range (0, len (tagged)):
            if tagged[w][1].startswith('J'):
                words[w] = lemma.lemmatize(tagged[w][0], wn.ADJ)
            elif tagged[w][1].startswith('V'):
                words[w] = lemma.lemmatize(tagged[w][0], wn.VERB)
            elif tagged[w][1].startswith('N'):
                words[w] = lemma.lemmatize(tagged[w][0], wn.NOUN)
            elif tagged[w][1].startswith('R'):
                words[w] = lemma.lemmatize(tagged[w][0], wn.ADV)
            else:
                words[w] = lemma.lemmatize(tagged[w][0])
        return " ".join(words)
    except:
        return post

# seems to be a better approach to only include true hits
def find_the_features (document, word_features):
    # all unique words in a document
    words = set (document)
    features = {}
    for w in words:
        if w in word_features:
            features[w] = True
        else:
            features[w] = False
    return features

def processInput (line: str) -> str:
    try:
        line = line.lower ()
    except AttributeError:
        pass
    line = stopWords (line)
    lemmatizer = WordNetLemmatizer ()
    line = lemmatizeMe (lemmatizer, line)
    return line

# Score by weights and output outcome
def handleClassification (weighted_features: dict, line: str) -> str:
    l_score, r_score = 0,0
    line_as_list = line.split()
    for x in line_as_list:
        if weighted_features.get(x) != None:
            if weighted_features[x][0].startswith('L'):
                l_score += float (weighted_features[x][1])
            else:
                r_score += float (weighted_features[x][1])
    if l_score > r_score:
        return "Liberal"
    else:
        return "Conservative"

# duplicated code as above to include/test additional sk-algos
def classify (weighted_features: dict, line: str) -> str:
    l_score, r_score = 0,0
    line_as_list = line.split()
    for x in line_as_list:
        if weighted_features.get(x) != None:
            if weighted_features[x][0].startswith('L'):
                l_score += float (weighted_features[x][1])
            else:
                r_score += float (weighted_features[x][1])
    if l_score > r_score:
        return f"Liberal: {l_score} > {r_score}"
    else:
        return f"Conservative: {l_score} < {r_score}"

# UNUSED 
def allWordsInList (words):
    all_words = []
    for post in words:
        try:
            words = word_tokenize (post.text)
            for w in words:
                if w not in string.punctuation:
                    all_words.append (w)
        except:
            pass
    return all_words

def allWordsInPost (post):
    all_words = []
    try:
        words = word_tokenize (post)
        for w in words:
            if w not in string.punctuation:
                all_words.append (w)
    except:
        pass   
    return all_words

### Phrase-based methods ###
def allWordPhrases (post, phrase_length: int):
    all_phrases = []
    phrase_length -= 1
    try:
        words = word_tokenize (post)
        for w in range (0, len(words)-phrase_length):
            if phrase_length == 0:
                all_phrases.append (
                f"{words[w]}")
            elif phrase_length == 1:
                all_phrases.append (
                f"{words[w]} {words[w+1]}")    
            elif phrase_length == 2:
                all_phrases.append (
                f"{words[w]} {words[w+1]} {words[w+2]}")
    except:
        pass   
    return all_phrases

# UNUSED 
# populate map with key = word, value (party, weight)
# returns 25 key words with weights
def updateWeightedFeatures (weighted_features: dict,
                            output: str, phrase_length: int) -> str:
    # return string
    highest_values = ""                         
    
    # polymorphic helper variables
    phrase_length -= 1
    loop_increment = 10 + phrase_length
    offset1toparty = 3 + phrase_length
    offset2toscore = 7 + phrase_length

    # format to list by delimiter, skip title
    output_as_list = output.split ()
    output_as_list = output_as_list[3:]
    # f = open ("weighted_features.txt", "a")
    for i in range (0, int (len (output_as_list) / loop_increment)):
        # polymorphic key depending on phrase size
        key = ""
        if phrase_length == 0:
            key = f"{output_as_list[i*loop_increment]}"
        elif phrase_length == 1:
            key = f"{output_as_list[i*loop_increment]} {output_as_list[i*loop_increment+1]}"
        elif phrase_length == 2:
            key = f"{output_as_list[i*loop_increment]} {output_as_list[i*loop_increment+1]} {output_as_list[i*loop_increment+2]}"
        
        # verify no overwriting of existing keys takes place
        if key not in weighted_features.keys ():
            # values 
            party = output_as_list[i*loop_increment+offset1toparty]
            score = output_as_list[i*loop_increment+offset2toscore]
            
            # update dictionary of phrases with score by dominant party
            weighted_features[key] = (party,score)

            # return the top 25 most decisive phrases
            if i < 25:
                highest_values += f"{key} : {party} : {score}\n"
    #     f.write (f"{i} : {key} : {output_as_list[i*12+5]} : {output_as_list[i*12+9]}\n")
    # f.close ()
    return highest_values
