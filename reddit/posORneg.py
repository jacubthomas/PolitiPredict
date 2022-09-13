# Enables us to import files outside this directory
import os
import sys
dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'nltk_tutorials'))
print (dir_path)
sys.path.insert(0, dir_path)

# INCLUDE THIS TO TRAIN CLASSIFIERS
import sentiment 

# INCLUDE THIS TO USE PRE-TRAINED CLASSIFIERS
# import sentiment_mod as sm

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
