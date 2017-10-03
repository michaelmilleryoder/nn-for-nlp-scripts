
# coding: utf-8

# In[1]:

import os
from subprocess import call


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

