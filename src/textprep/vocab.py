class Vocabulary:
    def __init__(
        self, add_unk, add_pad, 
        unk_token="<UNK>", pad_token="<PAD>", 
        word2index=None
    ) -> None:
        """
        Args:
            word2index (dict): a pre-­existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
            add_pad (bool): a flag that indicates whether to add the PAD token
            pad_token (str): the PAD token to add into the Vocabulary
        """

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
    def unk_index(self):
        return self._unk_index

    @property
    def word2index(self):
        return self._word2index

    @property
    def index2word(self):
        return self._index2word

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        if not isinstance(value, str):
            raise ValueError("[@_@] Required string parameter --> Passed was: {}".format(type(value)))
        self._pad_token = value

    @property
    def unk_token(self):
        return self._unk_token

    @unk_token.setter
    def unk_token(self, value):
        if not isinstance(value, str):
            raise ValueError("[@_@] Required string parameter --> Passed was: {}".format(type(value)))
        self._unk_token = value

    def add_token(self, token):
        """ Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """

        if token in self._word2index:
            index = self._word2index[token]
        else:
            index = len(self._word2index)
            self._word2index[token] = index
            self._index2word[index] = token

        return index

    def lookup_token(self, token):
        """ Retrieve the index associated with the token
            or the UNK index if token isn't present.
        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
            for the UNK functionality
        """

        if self._add_unk:
            return self.word2index.get(token, self.unk_index)
        else:
            return self.word2index[token]

    def lookup_index(self, index):
        """ Return the token associated with the index
        
        Args:
            index (int): the index to look up
        
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """

        if index not in self._index2word:
            raise KeyError("[@_@] index (%d) is not in the vocabulary" % index)

        return self.index2word[index]

    def __str__(self) -> str:
        contents = ""
        for word, i in self.word2index.items():
            contents += word + ": " + str(i) + "\n"
        return contents

    def __len__(self):
        return len(self.word2index)