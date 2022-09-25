import numpy as np


def load_embeddings_index(filepath):
    """ Load download word embeddings

    Args:
        filepath (str):the path to the file with the embeddings

    Returns:
        dict: a dictionary containing the embedding indices
    """
    embeddings_index = {}

    with open(filepath) as file_ref:
        for line in file_ref:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def create_embeddings_matrix(embeddings_index, word_index):
    """ Creates the embedding matrix which will be used in the model

    Args:
        embeddings_index (dict): word to vector dictionary
        word_index (dict): word to index dictionary
    """
    
    embedding_matrix = np.zeros((len(word_index) + 1, 100))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector