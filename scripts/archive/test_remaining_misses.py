import pandas as pd
import re

missed = pd.read_csv(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache\v61_missed_0.075.tsv', sep='\t')

def get_numeric_tokens(text):
    if not isinstance(text, str): return set()
    tokens = text.lower().split()
    return set([t for t in tokens if any(c.isdigit() for c in t)])

remaining = []
for idx, row in missed.iterrows():
    s1_nums = get_numeric_tokens(row['s1_text'])
    s2_nums = get_numeric_tokens(row['s2s3_text'])
    if len(s1_nums.intersection(s2_nums)) == 0:
        remaining.append(row)

remaining_df = pd.DataFrame(remaining)
print(f"Remaining Missed Pairs: {len(remaining_df)}")
remaining_df.to_csv(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache\v61_missed_remaining_0.075.tsv', sep='\t', index=False)
