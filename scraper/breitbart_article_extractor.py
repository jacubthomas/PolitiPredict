import time
import requests
from bs4 import BeautifulSoup

all_articles = set()
file = open('breitbart', 'r')
Lines = file.readlines()
i = 0
for line in Lines:
    if line.rstrip() not in all_articles:
        all_articles.add (line.rstrip())
        print (f'article {i}:::')
        response = requests.get(line.rstrip())
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        headlines.append(soup.find('h1').text)
        all = soup.find_all('p')
        for x in all:
            headlines.append(x.text)
        print (len (headlines))
        for x in headlines:
            print(x)
        i += 1

    print ('\n\n\n')
    time.sleep(1)