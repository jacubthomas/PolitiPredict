#!/Users/jacobharrington/opt/anaconda3/bin/python3

import os
import csv
import json
import time
import requests

def get_pushshift_data(after, before, sub):
    try:
        time.sleep (0.1)
        url = 'https://api.pushshift.io/reddit/search/submission/?after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
        print (url)
        r = requests.get(url)
        data = json.loads(r.text)
        return data['data']
    except:
        return None

# Get path to data directory
dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))

# open the file in the write mode
with open(dir_path+'/scrape_politics.csv', 'a') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write the header
    header = ['Title', 'Political_Party', 'Id', 'Subreddit', 'Date']
    writer.writerow(header)
    f.flush ()
    # last = 1656074696
    now = 1666560347
    # now = last
    onedecadeago = 1351052708
    subreddit = "politics"
    onehour = 3600
    start = now - onehour
    end = now
    data = get_pushshift_data(start, end, subreddit)
    errors, success = 0,0
    while start > onedecadeago:
        try:
            if data is not None:
                print (f'\ndata length: {len (data)}\n')
                if len (data) > 0:
                    for d in data:
                        post = [d['title'], 'Liberal', d['id'], subreddit, d['created_utc']]
                        print (f'{post}\n')
                        writer.writerow (post)
                        f.flush ()
            else:
                print ('data is none\n')
        except:
            # decrement now and before by one day - 86400, btter yet four hours
            start -= onehour
            end -= onehour
            errors += 1
            print (f'errors: {errors}, success: {success}\n')
            data = get_pushshift_data(start, end, subreddit)
            continue
        success += 1
        start -= onehour
        end -= onehour
        data = get_pushshift_data(start, end, subreddit)
    data = ['start: ' +str (start),
            'end: ' + str (end),
            'decade: ' + str (onedecadeago),
            'now: ' + str(now),
            '1 hour: ' + str (onehour)]
    writer.writerow (data)
    f.close ()