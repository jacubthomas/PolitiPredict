import requests
from bs4 import BeautifulSoup

all_articles = set()
file = open('msnbc_links', 'r')
Lines = file.readlines()
i = 0
for line in Lines:
    if line.rstrip() not in all_articles:
        all_articles.add (line.rstrip())
        print (f'article {i}:::')
        response = requests.get(line.rstrip())
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        headlines.append(soup.find('h1'))
        headlines.append(soup.find("div", class_="styles_articleDek__Icz5H"))
        all = soup.find_all('p')
        for x in all:
            headlines.append(x)
        for x in headlines:
            print(x.text.strip()) 
        i += 1

    print ('\n\n\n')