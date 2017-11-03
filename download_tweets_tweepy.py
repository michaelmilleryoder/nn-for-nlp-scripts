import os
import tweepy
import pandas as pd
import json
from tqdm import tqdm

# # Tweepy from status IDs

# OAuth
keys = pd.read_csv('tweepy_oath.txt', index_col=0)

auth = tweepy.OAuthHandler(keys.loc['consumer_key'], keys.loc['consumer_secret'])
auth.set_access_token(keys.loc['access_token'], keys.loc['access_secret'])

# Construct the API instance
api = tweepy.API(auth)


# Get ID list
data_dirpath = '/usr2/mamille2/twitter/data/huang2016_data'
train_fname = 'trainData_id.txt'
train_fpath = os.path.join(data_dirpath, train_fname)

train_data = pd.read_csv(train_fpath)

id_list = train_data['tweet_id'].tolist()


# Get tweet objects

print("Downloading tweets...")
for i in tqdm(range(len(id_list)//100 + 1)):
#for i in range(1):
    tweet_json_list = []
    tweets = api.statuses_lookup(id_list[i*100 : (i*100)+100])
    for t in tweets:
        tweet_json_list.append(t._json)
    
    # Write list out
    with open('/usr2/mamille2/twitter/data/huang2016_data/tweets/train{}.json'.format(i), 'w') as f:
        json.dump(tweet_json_list, f)
