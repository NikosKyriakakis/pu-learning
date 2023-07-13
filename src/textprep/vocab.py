from console import error


class Vocabulary:
    def __init__(
            self,
            add_unk: bool,
            add_pad: bool,
            unk_token: str = "<UNK>",
            pad_token: str = "<PAD>",
            word2index: dict = None
    ) -> None:

        if word2index is None:
            word2index = {}

        self._word2index = word2index
        self._index2word = {i: word for word, i in self._word2index.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self._pad_token = pad_token

        if add_pad:
            self.add_token(pad_token)

        self._unk_index = -1
        if add_unk:
            self._unk_index = self.add_token(unk_token)

    @property
    def unk_index(self) -> int:
        return self._unk_index

    @property
    def word2index(self) -> dict:
        return self._word2index

    @property
    def index2word(self) -> dict:
        return self._index2word

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError(error("Required string parameter --> Passed was: {}".format(type(value))))
        self._pad_token = value

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @unk_token.setter
    def unk_token(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError(error("Required string parameter --> Passed was: {}".format(type(value))))
        self._unk_token = value

    def add_token(self, token: str) -> int:
        if token in self._word2index:
            index = self._word2index[token]
        else:
            index = len(self._word2index)
            self._word2index[token] = index
            self._index2word[index] = token

        return index

    def lookup_token(self, token: str) -> int:
        if self._add_unk:
            return self.word2index.get(token, self.unk_index)
        else:
            return self.word2index[token]

    def lookup_index(self, index: int) -> str:
        if index not in self._index2word:
            raise KeyError(error("Index (%d) is not in the vocabulary" % index))

        return self.index2word[index]

    def __str__(self) -> str:
        contents = ""
        for word, i in self.word2index.items():
            contents += word + ": " + str(i) + "\n"
        return contents

    def __len__(self) -> int:
        return len(self.word2index)
