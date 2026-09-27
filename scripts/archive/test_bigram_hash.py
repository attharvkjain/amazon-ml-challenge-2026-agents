import pandas as pd
import re
import joblib, os, time

SHARED_CACHE = r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache'
data = joblib.load(os.path.join(SHARED_CACHE, 'train_data_1.pkl'))
s1 = data['val_s1']

def get_bigrams(text):
    if not isinstance(text, str): return set()
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return set([words[i] + ' ' + words[i+1] for i in range(len(words)-1)])

print("Extracting S1 bigrams...")
s1['bigrams'] = s1['name_address'].apply(get_bigrams)
bigram_counts = s1['bigrams'].explode().dropna().value_counts()

df = pd.read_csv(os.path.join(SHARED_CACHE, 'v5_missed_pairs_sample.tsv'), sep='\t')
df['s1_bigrams'] = df['s1_text'].apply(get_bigrams)
df['s23_bigrams'] = df['s2s3_text'].apply(get_bigrams)

for max_freq in [50, 100, 500, 1000]:
    rare_bigrams = set(bigram_counts[bigram_counts <= max_freq].index)
    def has_rare_overlap(row):
        overlap = row['s1_bigrams'].intersection(row['s23_bigrams'])
        return len(overlap.intersection(rare_bigrams)) > 0
    df['rare_overlap'] = df.apply(has_rare_overlap, axis=1)
    print(f'Bigram Max Freq <= {max_freq:4d}: Recovers {df.rare_overlap.mean()*100:.2f}% of missed pairs')
