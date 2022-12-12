# import requests
# from bs4 import BeautifulSoup

# url='https://www.bbc.com/news'
# url='https://www.foxnews.com/politics'
# url='https://www.msnbc.com/opinion/columnists'
# response = requests.get(url)

# soup = BeautifulSoup(response.text, 'html.parser')
# headlines = soup.find('body').find_all(class_='wide-tease-item__info-wrapper')
# for x in headlines:
#     print(x.text.strip())





# # MSN_BC
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('disable-notifications')
# driver = webdriver.Chrome('chromedriver.exe', options=chrome_options)


'''msnbc'''
# driver.get("https://www.msnbc.com/opinion/columnists")
# # print(driver.title)

# articles = driver.find_elements(By.TAG_NAME, "a")

# # print (len(articles))

# tabs = set()
# while (len(tabs) < 50):
#     try:
#         for x in articles:
#             link = x.get_attribute('href')
#             if link is not None and 'https://www.msnbc.com/opinion/msnbc-opinion' in link:
#                 tabs.add(link)           
#         button = driver.find_element(By.XPATH,'''//*[@id="content"]/div[6]/div/div[4]/div[1]/section/div/div[3]/button''')
#         button.click()
#         time.sleep (2)
#         articles = driver.find_elements(By.TAG_NAME, "a")
#     except:
#         time.sleep(4)
#         pop_up = driver.find_element(By.XPATH, '''//*[@id="sailthru-overlay-container"]/div[1]''')
#         pop_up.find_element(By.XPATH, '''//*[@id="sailthru-overlay-container"]/div[1]/button''').click()
#         time.sleep(2)
#         articles = driver.find_elements(By.TAG_NAME, "a")

# print (f'articles found : {len(tabs)}')
# for a in tabs:
#     print (a)

# time.sleep (15)
# driver.close()

'''Breitbart'''
# driver = webdriver.Chrome('./chromedriver')

# driver.get("https://www.breitbart.com/politics/")
# # print(driver.title)

# # articles = driver.find_elements(By.TAG_NAME, "article")
# articles = driver.find_elements(By.TAG_NAME, "a")
# page_on = 1
# tabs = set()
# while (len(tabs) < 65):
#     try:
#         for x in articles:
#             link = x.get_attribute('href')
#             if link is not None and 'https://www.breitbart.com/politics/20' in link:
#                 tabs.add(link)
#         if page_on == 1:
#             more_button = driver.find_element(By.XPATH,'//*[@id="MainW"]/nav/a')
#             more_button.click()
#         else: 
#             more_button = driver.find_element(By.XPATH,'//*[@id="MainW"]/nav/a[2]')
#             more_button.click()
#         for a in tabs:
#             print (a)
#         articles = driver.find_elements(By.TAG_NAME, "a")
#         page_on += 1
#     except:
#         time.sleep(2)
#         try:
#             pop_up2 = driver.find_element(By.XPATH, '''//*[@id="IL_SR_X4"]/svg/g/path[1]''')
#             pop_up2.click()
#         except: 
#             pass
# print (f'articles found : {len(tabs)}')
# for a in tabs:
#     print (a)

# time.sleep (15)
# driver.close()

'''huffpost'''
tabs = set()
for i in range(1,21):
    driver = webdriver.Chrome('chromedriver.exe', options=chrome_options)
    driver.get(f"https://www.washingtontimes.com/news/politics/?page={i}")
    articles = driver.find_elements(By.TAG_NAME, "a")
    try:
        for x in articles:
            link = x.get_attribute('href')
            if link is not None and 'https://www.washingtontimes.com/news/2022/' in link:
                tabs.add(link)
    except:
        pass
    driver.close()
    time.sleep(1)

for x in tabs:
    print (x)