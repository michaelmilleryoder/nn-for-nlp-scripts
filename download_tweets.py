import os
from subprocess import call
import tqdm

# Download tweets from the tweet ids provided by TREC2011
trec_dir = 'trec2011/'
app_path = 'twitter-tools/twitter-tools-core/target/appassembler/bin/AsyncHTMLStatusBlockCrawler'

date_dirs = sorted([name for name in os.listdir(trec_dir) if os.path.isdir(os.path.join(trec_dir, name))])

for date_dir in tqdm(date_dirs, desc="directories"):
    print(date_dir)
    date_dirpath = os.path.join(trec_dir, date_dir)
    
    dat_files = sorted([fname for fname in os.listdir(date_dirpath) if fname.endswith('.dat')])

    for dat_fname in tqdm(dat_files, desc="files"):
        out_path = os.path.join(trec_dir, date_dirpath, 'json', dat_fname[:-3] + 'json.gz')
        dat_path = os.path.join(trec_dir, date_dir, dat_fname)
        print("Processing {}...".format(dat_path))
        call([app_path, '-data', dat_path, '-output', out_path])
        print("\nOutput written to {}".format(out_path))
    
    print()

