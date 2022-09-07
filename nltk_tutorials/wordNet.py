from nltk.corpus import wordnet as wn

# synsets
trust = wn.synsets ("trust")

# all synonyms, all lemmas of synonym, print name
for t in trust:
	for l in t.lemmas ():
		print (f"{l.name ()}, ", end='')
	print ()





