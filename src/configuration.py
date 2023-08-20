import gdown
import json
import os, sys
import subprocess

from console import error, hourglass, success, warning
from zipfile import ZipFile


def load_settings(filename: str) -> dict:
    print(hourglass("Loading project settings"))
    with open(filename) as file_stream:
        settings = json.load(file_stream)

    return settings


def extract_file(filename: str, output_dir: str) -> None:
    if os.name == "nt":
        filename = filename.replace("/", "\\")
    print(hourglass("Extracting file: {}".format(filename)))
    with ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(path=output_dir)


class DownloadManager:
    def __init__(
            self,
            settings
    ) -> None:

        self.settings = settings
        self.restore_path = os.getcwd()

    def download_embeddings(self, option: str, force_extraction: bool = False) -> None:
        print(warning(f"Attempting to download {option} ..."))
        embedding_dir = self.settings["embedding_dir"]
        embedding_options = self.settings["embedding_options"]

        if not os.path.exists(embedding_dir):
            os.mkdir(embedding_dir)

        if option not in embedding_options:
            raise UserWarning(
                error("Unknown embedding option provided --> Supported files: {}".format(embedding_options.keys()))
            )

        os.chdir(embedding_dir)

        embedding = embedding_options[option]
        filename = os.path.join(embedding_dir, embedding["filename"])

        output = os.path.join(embedding_dir, option)
        dest_exists = os.path.exists(output)
        if not dest_exists or force_extraction:
            if os.path.exists(filename):
                print(success("Embedding file already present, skipping download ..."))
            else:
                try:
                    command = embedding["command"].split()
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print("Execution failed:", e, file=sys.stderr)
                    sys.exit(-1)

            extract_file(filename, output_dir=output)

        os.chdir(self.restore_path)

    def download_from_gdrive(self, resource: str) -> None:
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
