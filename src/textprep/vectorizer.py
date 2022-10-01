from textprep.vocab import Vocabulary
from nltk.stem import WordNetLemmatizer

import pandas as pd
import string


lemmatizer = WordNetLemmatizer()


def to_lower(document):
    return document.lower().strip()


def to_remove_symbols(document):
    processed = ""
    for letter in document:
        if letter not in string.punctuation:
            processed += letter
    return processed


class Vectorizer:
    def __init__(self, text_vocab, label_vocab) -> None:
        self._text_vocab = text_vocab
        self._label_vocab = label_vocab

    @property
    def text_vocab(self):
        return self._text_vocab

    @property
    def label_vocab(self):
        return self._label_vocab

    @classmethod
    def from_dataframe(cls, data, text_column, label_column, prep_funcs={}):
        """ Instantiate a vectorizer from the passed in dataframe

        Args:
            data (pandas.DataFrame): the data to vectorize
        Returns: 
            an instance of TextVectorizer
        """

        if type(data) != pd.DataFrame:
            raise ValueError("[o_O] Illegal parameter provided --> 'data' argument should be a pandas DataFrame")
        
        if text_column not in data.columns:
            raise ValueError("[o_O] Provided text column was not found in dataframe's columns")

        if label_column not in data.columns:
            raise ValueError("[o_O] Provided label column was not found in dataframe's columns")

        text_vocab = Vocabulary(add_unk=True, add_pad=True)
        label_vocab = Vocabulary(add_unk=False, add_pad=False)

        for label in sorted(set(data[label_column])):
            label_vocab.add_token(label)

        sequences = []
        documents = []
        max_len = 0
        for document in data.text:
            for func, params in prep_funcs.items():
                params["document"] = document
                document = func(**params)
            documents.append(document)

            sequence = []
            for word in document.split():
                word_index = text_vocab.add_token(word)
                sequence.append(word_index)

            current_len = len(sequence)
            if current_len > max_len:
                max_len = current_len

            sequences.append(sequence)

        pad_token = text_vocab.pad_token
        pad_index = text_vocab.lookup_token(pad_token)
        for i in range(len(sequences)):
            pad_size = max_len - len(sequences[i])
            padding = pad_size * [pad_index]
            sequences[i] += padding

        vectorizer = Vectorizer(text_vocab, label_vocab)

        return sequences, documents, vectorizer