import json
from file_read_backwards import FileReadBackwards


def save_logs(output):
    with open("report.log", "a") as checkpoint:
        json.dump(output, checkpoint, indent=4)
        checkpoint.write(",\n")