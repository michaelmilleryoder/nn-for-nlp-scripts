
# coding: utf-8

# In[15]:

import os
from subprocess import call
import tweepy
import pandas as pd
import json


# # Tweepy from status IDs

# In[2]:

# OAuth
auth = tweepy.OAuthHandler("GsoM6eLDx8SjfrkV277vjtxEE", "qgM9eMqj7E8onWbhLddHZyhd8r67S4fYaCILILS291zeYg0OAn")
auth.set_access_token('926413946785091584-Ynm0D6w6VtlCr3yMs9hiGSRhMIjp9om', 'Iir8vn4r8fxewkRmihlyLfnc2IMaDCBbJ7UBIgEB5nd4S')

# Redirect user to Twitter to authorize
# redirect_user(auth.get_authorization_url())

# Get access token
# auth.get_access_token("verifier_value")

# Construct the API instance
api = tweepy.API(auth)


# In[3]:

# Get ID list
data_dirpath = '/home/michael/school/11-747/huang2016_data'
train_fname = 'trainData_id.txt'
train_fpath = os.path.join(data_dirpath, train_fname)

train_data = pd.read_csv(train_fpath)
train_data


# In[4]:

id_list = train_data['tweet_id'].tolist()
len(id_list)


# In[5]:

tweet_json_list = []
tweets = api.statuses_lookup(id_list[:10])
for t in tweets:
    tweet_json_list.append(t._json)
    
len(tweet_json_list)


# In[16]:

with open('/home/michael/school/11-747/test_twitter.json', 'w') as f:
    json.dump(tweet_json_list, f)


# In[17]:

with open('/home/michael/school/11-747/test_twitter.json', 'r') as f:
    test_list = json.load(f)
    
test_list == tweet_json_list


# # TREC2011

# In[2]:

trec_dir = '/home/michael/school/11-747/trec2011/'


# In[16]:

date_dirs = sorted([name for name in os.listdir(trec_dir) if os.path.isdir(os.path.join(trec_dir, name))])

for date_dir in date_dirs[:1]:
    print(date_dir)
    date_dirpath = os.path.join(trec_dir, date_dir)
    
    dat_files = sorted([fname for fname in os.listdir(date_dirpath) if fname.endswith('.dat')])
    for dat_fname in dat_files[1:2]:
        app_path = '/home/michael/software/twitter-tools/twitter-tools-core/target/appassembler/bin/AsyncHTMLStatusBlockCrawler'
        out_path = os.path.join(trec_dir, date_dirpath, 'json', dat_fname[:-3] + 'json.gz')
        dat_path = os.path.join(trec_dir, date_dir, dat_fname)
        print("Processing {}...".format(dat_path))
        call([app_path, '-data', dat_path, '-output', out_path])
        print("\nOutput written to {}".format(out_path))
    
    print()

