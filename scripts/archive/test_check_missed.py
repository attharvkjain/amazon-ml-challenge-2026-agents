import pandas as pd
import re

missed = pd.read_csv(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache\v61_missed_0.075.tsv', sep='\t')

def get_numeric_tokens(text):
    if not isinstance(text, str): return set()
    tokens = text.lower().split()
    nums = [t.lstrip('0') for t in tokens if any(c.isdigit() for c in t)]
    return set([t for t in nums if len(t) > 0])

recovered = 0
for idx, row in missed.iterrows():
    s1_nums = get_numeric_tokens(row['s1_text'])
    s2_nums = get_numeric_tokens(row['s2s3_text'])
    shared = s1_nums.intersection(s2_nums)
    if len(shared) > 0:
        recovered += 1
        
print(f"Total Missed: {len(missed)}")
print(f"Pairs sharing numeric token: {recovered}")
