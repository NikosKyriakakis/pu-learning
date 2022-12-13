import gdown, json, os

from zipfile import BadZipFile, ZipFile
from console import *


def load_settings(filename):
    settings = {}

    try:
        with open(filename) as file_stream:
            settings = json.load(file_stream)
    except IOError as io_error:
        raise UserWarning(error("{}".format(io_error)))
    
    return settings


def extract_file(filename, output_dir):
    """ Extract all contents of the provided file

    Args:
        filename (str): the name of the file to extract
        output_dir (str): the destination folder 
    """

    try:
        print(hourglass("Extracting file: {}".format(filename)))
        with ZipFile(filename, "r") as zipref:
            zipref.extractall(path=output_dir)
    except BadZipFile as bad_zip:
        raise UserWarning(error("{}".format(bad_zip)))
    except IOError as io_error:
        raise UserWarning(error("{}".format(io_error)))


class DownloadManager:
    def __init__(
        self, 
        settings
    ) -> None:
    
        self.settings = settings
        self.restore_path = os.getcwd()

    def download_embeddings(self, option, force_extraction=False):
        """ Download pretrained embeddings

        Args:
            option (str): the name of the embedding to download
            force_extraction (bool): a boolean flag to determine whether to override any existing folder
        
        Raises:
            UserWarning when an unknown option is provided 
        """

        embedding_dir = self.settings["embedding_dir"]
        embedding_options = self.settings["embedding_options"]

        if not os.path.exists(embedding_dir):
            os.mkdir(embedding_dir)

        if option not in embedding_options:
            raise UserWarning(error("Unknown embedding option provided --> Supported files: {}".format(embedding_options.keys())))

        os.chdir(embedding_dir)

        embedding = embedding_options[option]
        filename = os.path.join(embedding_dir, embedding["filename"])

        output = os.path.join(embedding_dir, option)
        dest_exists = os.path.exists(output)
        if not dest_exists or force_extraction == True:
            if os.path.exists(filename):
                print(success("Embedding file already present, skipping download ..."))
            else:
                os.system(embedding["command"])
            extract_file(filename, output_dir=output)

        os.chdir(self.restore_path)

    def download_from_gdrive(self, resource):
        """ Download available datasets from remote Google drive account

        Args:
            resource (str): the name of the associated file stored in the Drive
            destination (str, optional): the path where the data folder exists. Defaults to "../../data/".

        Raises:
            UserWarning when an unknown option is provided 
        """

        dataset_options = self.settings["dataset_options"]
        data_dir = self.settings["data_dir"]

        if resource in dataset_options:
            url = dataset_options[resource]
        else:
            raise UserWarning(error("Selected resource is not available."))
        
        if not os.path.exists(data_dir):
            print(warning("Provided destination folder does not exist --> Creating one now ..."))
            os.mkdir(data_dir)

        output = os.path.join(data_dir, resource)
        
        if os.path.exists(output):
            print(success("Dataset {} found locally --> Aborting download ...".format(resource)))
            return 

        gdown.download(url, output, quiet=False)