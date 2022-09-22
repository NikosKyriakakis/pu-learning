import numpy as np
import gdown
import sys
import os


def project_setup():
    """ Create necessary project folders """

    # Create data folder if absent
    if not os.path.exists("../data/"):
        os.mkdir("../data/")
    # Create embeddings folder if empty
    if not os.path.exists("../embeddings/"):
        os.mkdir("../embeddings/")


def download_from_gdrive(resource, destination = "../data/"):
    """ Download available datasets from remote Google drive account

    Args:
        resource (str): the name of the associated file stored in the Drive
        destination (str, optional): the path where the data folder exists. Defaults to "../../data/".
    """
    remote_resources = {
        "deceptive-opinion.csv": "https://drive.google.com/uc?id=1QaV8r3l4EohQACCiwORr6Hqb9HQVhXV7"
    }

    if resource in remote_resources:
        url = remote_resources[resource]
    else:
        print("[o_O] FATAL: Selected resource is not available.")
        sys.exit(-1)
    
    if not os.path.exists(destination):
        print("[~_o] Provided destination folder does not exist --> Creating one now ...")
        os.mkdir(destination)

    output = os.path.join(destination, resource)
    
    if os.path.exists(output):
        print("[~_o] File already exists --> Aborting download")
        return 

    gdown.download(url, output, quiet=False)


def load_embeddings(path):
    embeddings_index = {}

    with open(path, "r") as FILE:
        for line in FILE:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    return embeddings_index


def create_embeddings_matrix(embeddings_index, word_index, max_len=100):
    embedding_matrix = np.zeros((len(word_index) + 1, max_len))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix