import os
import logging
import pandas as pd
import numpy as np

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, size=0, batch_size=1):
        self.size = size
        self.batch_size = batch_size
        if size == 0:
            self.total = 0
        else:
            self.total = np.zeros((1, size))
        self.steps = 0

    def update(self, val):
        if self.batch_size == 1 or val.shape[0] == 1:
            self.total += val
            self.steps += 1
        else:
            for i in range(val.shape[0]):
                self.total += val[i]
                self.steps += 1

    def __call__(self):

        if self.size == 0:
            return self.total / float(self.steps)
        else:

            return np.mean(self.total / float(self.steps), axis=0)


def set_logger(log_path, override=False):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is
    saved in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
        override (bool): choose to over ride or not, only will override if true is passed
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if override:
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
    if not logger.handlers or override:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(stream_handler)


def convert_excel_to_csv(file, sheet=None, outname=None):
    """ function that converts data in excel sheet into a csv format

    Args:
        file (str): excel file that we are converting
        sheets (none): gets all worksheets
    Return:
        outname (none): file we are saving info to as csv
    """

    data = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")
    if outname is not None:
        data.to_csv(outname)
    else:
        outname = file.replace(".xlsx", ".csv")
        data.to_csv(outname)
    print("Converted " + file + " to " + outname)
    return outname


def touch(fname, times=None):
    """from [stackoverflow](https://stackoverflow.com/questions/1158076/implement-touch-using-python) # noqa

    Args:
        fname (str): string of filename 
    Optional:
        times (none): time which defaults to None.
    """
    with open(fname, "a"):
        os.utime(fname, times)
