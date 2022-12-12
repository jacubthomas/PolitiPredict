import os
import csv
import praw

reddit = praw.Reddit (client_id = '44O6Yoza4MDf3vUfeuGY8Q',
                      client_secret = 'jpgQeigu2KS00YBIDooCcTZQnFu1cA',
                      username = 'Commercial_Edge_5378',
                      password = ':FZA_@3HH5aV3u+',
                      user_agent = 'prawtutorialv1')


dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))

# open the file in the write mode
# with open(dir_path+'/scrape_liberal.csv', 'w') as f:
with open(dir_path+'/scrape_republican2.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write the header
    header = ['Title', 'Political_Party', 'Id', 'Subreddit', 'Date']
    writer.writerow(header)

    sub = 'Republican'
    subreddit = reddit.subreddit (sub)

    hot_python = subreddit.top(limit=10000)

    for submission in hot_python:
        # if not submission.stickied:
        data = [submission.title, 'Conservative', submission.id, sub, submission.created_utc]
        writer.writerow (data)
