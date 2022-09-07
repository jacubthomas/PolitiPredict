# pip3 install nltk	# DONE

# import nltk		# DONE

# nltk.download()       # DONE


# tokenizing - word tokenizers .... sentence tokenizers

# lexicon and corporas
# corpora - body of text. ex: medical journals, presidential speeches, English langugae
# lexicon - words and their meanings ( variation within )
#	investor-speak ..... regular english-speak
	# investor 'bull' positive about market
	# english 'bull' scary animal


from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python3 is awesome. The sky is pinkish-blue. You should not eat cardboard."

print (example_text)

print ()

print (sent_tokenize (example_text) )

print ()

print (word_tokenize (example_text) )

print ()

for i in word_tokenize (example_text):
	print (i)
