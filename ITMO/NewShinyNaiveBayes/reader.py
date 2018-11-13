import os

from pprint import pprint

SPAM_LABEL = "spmsg"


class Mail:
    """
    Class for mail

    init with:
        subject: List[int]
        body: List[int]
        is_spam: Bool (default = False)
    """

    def __init__(self, subject, body, title, is_spam=False):
        self.title = title
        self.subject = subject
        self.body = body
        self._is_spam = is_spam

    @property
    def spam(self):
        return self._is_spam

    @property
    def ham(self):
        return not self._is_spam

    def __repr__(self):
        return f'{self.title} - {self.spam}\n\t{self.subject}\n\t{self.body}'

    def __str__(self):
        return self.__repr__()


class DataFold:

    def __init__(self, hams, spams):
        self.hams = hams
        self.spams = spams

    def __repr__(self):
        hams_repr = '\n'.join([str(m)for m in self.hams])
        spams_repr = '\n'.join([str(m)for m in self.spams])
        return f'hams: {hams_repr}\n\t-----\nspams: {spams_repr}'

    def __str__(self):
        return self.__repr__()


def read_dataset_folds(dir_path):
    folds = []
    for dir in os.listdir(dir_path):
        messages = _read_part(os.path.join(dir_path, dir))
        hams = [message for message in messages if message.ham]
        spams = [message for message in messages if message.spam]
        folds.append(DataFold(hams, spams))
    return folds


def _read_part(dir_path):
    messages = []
    for file_path in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_path)) as file_obj:
            subject, body = _read_message(file_obj)
            is_spam = False if file_path.find(SPAM_LABEL) < 0 else True
            messages.append(Mail(subject, body, file_path, is_spam))
    return messages


def _read_message(file_obj):
    all_text = file_obj.read().split('\n')
    assert len(all_text) == 4
    subject, _, body, _ = all_text

    subject = subject.strip().split(' ')[1:]
    body = body.strip().split(' ')

    return subject, body

# TEST
# if __name__ == '__main__':
#     folds = read_dataset_folds(r'')
#     pprint(folds[0])
