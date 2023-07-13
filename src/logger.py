import json
from file_read_backwards import FileReadBackwards


def save_logs(output):
    with open("report.log", "a") as checkpoint:
        json.dump(output, checkpoint, indent=4)
        checkpoint.write(",\n")


def reset_logging(output="EXPERIMENT START"):
    with open("report.log", "a") as checkpoint:
        checkpoint.write("\n" + 45 * "=" + " " + output + " " + 45 * "=" + "\n")
