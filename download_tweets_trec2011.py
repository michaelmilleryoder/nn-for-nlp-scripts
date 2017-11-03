import os
import subprocess
from tqdm import tqdm

# Download tweets from the tweet ids provided by TREC2011
trec_dir = 'twitter/data/trec2011/'
app_path = 'twitter/twitter-tools/twitter-tools-core/target/appassembler/bin/AsyncHTMLStatusBlockCrawler'

date_dirs = sorted([name for name in os.listdir(trec_dir) if os.path.isdir(os.path.join(trec_dir, name))])

#for date_dir in tqdm(date_dirs, desc="directories"):
for date_dir in date_dirs: # are 17
    tqdm.write(date_dir)
    date_dirpath = os.path.join(trec_dir, date_dir)
    
    dat_files = sorted([fname for fname in os.listdir(date_dirpath) if fname.endswith('.dat')])

    if not os.path.exists(os.path.join(date_dirpath, 'json')):
        os.makedirs(os.path.join(date_dirpath, 'json'))

    #for dat_fname in tqdm(dat_files, desc="files"):
    for dat_fname in tqdm(dat_files):
        out_path = os.path.join(date_dirpath, 'json', dat_fname[:-3] + 'json.gz')
        dat_path = os.path.join(date_dirpath, dat_fname)
        #tqdm.write("Processing {}...".format(dat_path))
        subprocess.call([app_path, '-data', dat_path, '-output', out_path], stdout=subprocess.DEVNULL)
        #subprocess.call([app_path, '-data', dat_path, '-output', out_path])
            # piping output to suppress

        #tqdm.write("\tOutput written to {}".format(out_path))
    
    #tqdm.write()

