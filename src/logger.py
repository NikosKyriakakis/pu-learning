import json
import os

from Crypto.Hash import MD5


def hash_text(text):
    hash = MD5.new()
    hash.update(text.encode('utf-8'))
    return hash.hexdigest()


def save_logs(output, destination_folder="../reports"):
    restore_path = os.getcwd()
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    os.chdir(destination_folder)
    hash = hash_text(json.dumps(output))

    with open(f"report_{hash}.json", "w") as checkpoint:
        json.dump(output, checkpoint, indent=4)

    os.chdir(restore_path)