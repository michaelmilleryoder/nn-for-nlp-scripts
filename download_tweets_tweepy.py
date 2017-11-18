import os
import tweepy
import pandas as pd
import json
from tqdm import tqdm
import pdb

# # Tweepy from status IDs

# OAuth
keys = pd.read_csv('/usr2/mamille2/twitter/tweepy_oauth.txt', index_col=0)

auth = tweepy.OAuthHandler(keys.loc['consumer_key', 'key'], keys.loc['consumer_secret', 'key'])
auth.set_access_token(keys.loc['access_token', 'key'], keys.loc['access_secret', 'key'])

# Construct the API instance
api = tweepy.API(auth)


# Get ID list
data_dirpath = '/usr2/mamille2/twitter/data/huang2016_data'
fold = 'test'
outdir = os.path.join(data_dirpath, 'tweets', fold)
data_fname = '{}Data_id.txt'.format(fold)
data_fpath = os.path.join(data_dirpath, data_fname)

data = pd.read_csv(data_fpath)

id_list = data['tweet_id'].tolist()
print("Saw {} tweets".format(len(id_list)))


# Get tweet objects

print("Downloading tweets...")
for i in tqdm(range(len(id_list)//100 + 1)):

    outpath = os.path.join(outdir, '{}{:04}.json'.format(fold, i))

    if os.path.isfile(outpath):
        continue

    tweet_json_list = []

    try:
        tweets = api.statuses_lookup(id_list[i*100 : (i*100)+100])

    except tweepy.TweepError as e:
        print(e)
        continue

    for t in tweets:
        tweet_json_list.append(t._json)
    
    # Write list out
    with open(outpath, 'w') as f:
        json.dump(tweet_json_list, f)
