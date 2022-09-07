from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer ()

print (lemmatizer.lemmatize ("cats"))
print (lemmatizer.lemmatize ("cacti")) 
print (lemmatizer.lemmatize ("geese"))
print (lemmatizer.lemmatize ("rocks")) 
print (lemmatizer.lemmatize ("python")) 
print (lemmatizer.lemmatize ("rocky"))

# Default parameter for lemmatize, pos = "n" noun
print (lemmatizer.lemmatize ("better"))
print (lemmatizer.lemmatize ("better", pos="a"))
print (lemmatizer.lemmatize ("best", pos="a"))

print (lemmatizer.lemmatize ("run"))
print (lemmatizer.lemmatize ("run", pos="v"))
