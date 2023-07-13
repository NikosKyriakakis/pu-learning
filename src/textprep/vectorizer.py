import pandas as pd
import string

from console import error
from textprep.vocab import Vocabulary


def to_lower(document: str) -> str:
    return document.lower().strip()


def to_remove_symbols(document: str) -> str:
    processed = ""
    for letter in document:
        if letter not in string.punctuation:
            processed += letter
    return processed


class SequenceVectorizer:
    def __init__(self, text_vocab: Vocabulary, label_vocab: Vocabulary, max_len: int) -> None:
        self._text_vocab = text_vocab
        self._label_vocab = label_vocab
        self.max_len = max_len

    @property
    def text_vocab(self) -> Vocabulary:
        return self._text_vocab

    @property
    def label_vocab(self) -> Vocabulary:
        return self._label_vocab

    @classmethod
    def from_dataframe(
            cls,
            data: pd.DataFrame,
            text_column: str = "text",
            label_column: str = "label",
            prep_funcs: dict = None
    ):
        if prep_funcs is None:
            prep_funcs = {}

        if type(data) != pd.DataFrame:
            raise ValueError(error("Illegal parameter provided --> 'data' argument should be a pandas DataFrame"))

        if text_column not in data.columns:
            raise ValueError(error("Provided text column was not found in dataframe's columns"))

        if label_column not in data.columns:
            raise ValueError(error("Provided label column was not found in dataframe's columns"))

        text_vocab = Vocabulary(add_unk=True, add_pad=True)
        label_vocab = Vocabulary(add_unk=False, add_pad=False)

        for label in sorted(set(data[label_column])):
            label_vocab.add_token(label)

        max_len = 0
        for idx, document in enumerate(data[text_column]):
            for func, params in prep_funcs.items():
                params["document"] = document
                document = func(**params)
            data[text_column][idx] = document

            words = document.split()
            for word in words:
                text_vocab.add_token(word)

            current_len = len(words)
            if current_len > max_len:
                max_len = current_len

        vectorizer = SequenceVectorizer(text_vocab, label_vocab, max_len)

        return vectorizer

    def vectorize(self, document: str) -> list[int]:
        pad_token = self.text_vocab.pad_token
        pad_index = self.text_vocab.lookup_token(pad_token)

        sequence = []
        for word in document.split():
            word_index = self.text_vocab.lookup_token(word)
            sequence.append(word_index)

        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]

        pad_size = self.max_len - len(sequence)
        padding = pad_size * [pad_index]
        sequence += padding

        return sequence
