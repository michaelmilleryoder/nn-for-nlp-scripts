import os
import tweepy
import pandas as pd
import json
from tqdm import tqdm
import pdb
import time

# # Tweepy from user IDs

# OAuth
keys = pd.read_csv('/usr2/mamille2/twitter/tweepy_oauth.txt', index_col=0)

auth = tweepy.OAuthHandler(keys.loc['consumer_key', 'key'], keys.loc['consumer_secret', 'key'])
auth.set_access_token(keys.loc['access_token', 'key'], keys.loc['access_secret', 'key'])

# Construct the API instance
api = tweepy.API(auth)


# Get user ID list
data_dirpath = '/usr2/mamille2/twitter/data/huang2016_data'

for fold in ['valid', 'test']:

    print(fold)

    data_fname = '{}_uids.txt'.format(fold)
    data_fpath = os.path.join(data_dirpath, data_fname)
    outdir = os.path.join(data_dirpath, 'user_histories', fold)

    with open(data_fpath) as f:
        uids = f.read().splitlines()

    print("See {} users".format(len(uids)))


    # Get tweet objects

    print("Downloading tweets...")
    for i in tqdm(range(len(uids)//100 + 1)):

        outpath = os.path.join(outdir, '{}{:03}.json'.format(fold, i))

        if os.path.isfile(outpath):
            continue

        tweet_json_list = []

        for j in uids[i*100 : (i*100)+100]:
            try:
                tweets = api.user_timeline(user_id=j, count=5)

            except tweepy.TweepError as e:
                
                if isinstance(e, tweepy.error.RateLimitError):
                    tqdm.write("\tSleeping 5 min...")
                    time.sleep(60*5)
                    continue
            
                else:
                    print(e)
                    continue

            for t in tweets:
                tweet_json_list.append(t._json)
        
        # Write list out
        with open(outpath, 'w') as f:
            json.dump(tweet_json_list, f)
