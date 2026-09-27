import os
import sys
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SUBMISSION', 'code', 'business_entity_resolution', 'src'))
from blocking.heuristic_blocker import _generate_numeric_candidates

data = joblib.load(r'F:\PROJECTS\AmazonMLChallenge_2026\SUBMISSION\code\business_entity_resolution\cache\train_data_0.075.pkl')

s1 = data['val_s1'].head(1000)
s2 = data['val_s2'].head(1000)

res = _generate_numeric_candidates(s1, s2)
print("Result rows:", len(res))
