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
        "glove": ("wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip"),
        "fasttext-wiki": ("wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip", "wiki-news-300d-1M.vec.zip"),
        "fasttext-crawl": ("wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip", "crawl-300d-2M.vec.zip")
    }

    if option not in embedding_options:
        print("[@_@] Unknown embedding option provided --> Supported files: {}".format(embedding_options.keys()))
        return

    os.chdir(output_dir)

    command, zip_file = embedding_options[option]
    filename = os.path.join(output_dir, zip_file)
    if os.path.exists(filename):
        print("[~_o] Embedding file already present, skipping download ...")
    else:
        os.system(command)

    output = os.path.join(output_dir, option)
    extract_file(filename, output_dir=output)


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
        print("[~_o] Dataset already exists --> Aborting download")
        return 

    gdown.download(url, output, quiet=False)