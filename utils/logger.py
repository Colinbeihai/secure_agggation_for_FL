import datetime
import os

class Logger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

    def log(self, msg):
        time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{time} {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\\n")
