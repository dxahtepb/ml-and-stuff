import sys
import os


class Mail:
    """
    Class for mail

    init with:
        subject: List[int]
        body: List[int]
        is_spam: Bool (default = False)
    """

    def __init__(self, subject, body, is_spam=False):
        self.subject = subject
        self.body = body
        self._is_spam = is_spam

    @property
    def spam(self):
        return self._is_spam

    @property
    def ham(self):
        return not self._is_spam


class DataFold:

    def __init__(self, hams, spams):
        self.hams = hams
        self.spams = spams


def read_dataset_folds(dir_path):
    folds = []
    for dir in os.listdir(dir_path):
        folds.append(_read_part(os.path.join(dir_path, dir)))
    return folds


def _read_part(dir_path):
    for file_path in os.listdir(dir_path):
        with open(file_path) as file_obj:
            pass
    return


def _read_message():
    pass