import time
import requests
from bs4 import BeautifulSoup

all_articles = set()
for i in range(2,21):
    response = requests.get(f'https://www.huffpost.com/news/politics?page={i}')
    soup = BeautifulSoup(response.text, 'html.parser')
    all = soup.find_all("a", href=True)
    for x in all:
        if 'https://www.huffpost.com/entry/' in x['href']:
            all_articles.add(x['href'])

i = 0
for x in all_articles:
    print (f'article {i}:::')
    response = requests.get(x)
    soup = BeautifulSoup(response.text, 'html.parser')
    for para in soup.find_all("p"):
        print(para.get_text())
    i += 1