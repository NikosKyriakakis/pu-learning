import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def prepare(row):
    filtered = []
    row = row.lower().strip()
    row = re.sub('[^A-Za-z0-9.]+', ' ', row)
    row_parts = row.split()
    for part in row_parts:
        filtered.append(lemmatizer.lemmatize(part))
    return filtered