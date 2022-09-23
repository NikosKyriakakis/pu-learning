import numpy as np
import gdown
import sys
import os


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