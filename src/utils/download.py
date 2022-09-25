import gdown
import sys
import os

from zipfile import BadZipFile, ZipFile

def extract_file(filename, output_dir):
    """ Extract all contents of the provided file

    Args:
        filename (str): the name of the file to extract
        output_dir (str): the destination folder 
    """

    try:
        with ZipFile(filename, "r") as zipref:
            zipref.extractall(path=output_dir)
    except BadZipFile as bad_zip:
        print("[o_O] {}".format(bad_zip))
        return 1

    return 0


def download_embeddings(option):
    """ Download pretrained embeddings

    Args:
        option (str): the name of the embedding to download

    Returns:
        int: status code
    """
    output_dir = "../embeddings/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    embedding_options = {
        "glove": ("wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip -O /tmp/glove.6B.zip", "/tmp/glove.6B.zip")
    }

    if option not in embedding_options:
        print("[@_@] Unknown embedding option provided --> Supported files: {}".format(embedding_options.keys()))
        return 1
    
    command, filepath = embedding_options[option]
    if os.path.exists(filepath):
        print("[@_@] Embedding file already present, skipping download ...")
    else:
        os.system(command)


    output = os.path.join(output_dir, option)
    if not os.path.exists(output):
        os.mkdir(output)

    extract_file(filepath, output_dir=output)

    return 0


def download_from_gdrive(resource, destination="../data/"):
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