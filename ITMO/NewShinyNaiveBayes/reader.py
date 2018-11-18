import os


SPAM_LABEL = "spmsg"


class Mail:
    """
    Class for mail

    init with:
        subject: List[int]
        body: List[int]
        is_spam: Bool (default = False)
    """

    def __init__(self, subject, body, title, fold, is_spam=False):
        self.title = title
        self.subject = set(subject)
        self.body = set(body)
        self.all_words = sorted(list(set(self.subject.union(self.body))))
        self.subject = sorted(list(set(self.subject)))
        self.body = sorted(list(set(self.body)))
        self._is_spam = is_spam
        self.fold = fold

    @property
    def spam(self):
        return self._is_spam

    @property
    def ham(self):
        return not self._is_spam

    def __repr__(self):
        return '{} - {}\n\t{}\n\t{}'.format(
            self.title, self.spam, self.subject, self.body)

    def __str__(self):
        return self.__repr__()


def read_dataset_folds(dir_path):
    folds = []
    for fold_n, dir in enumerate(os.listdir(dir_path)):
        messages = _read_part(os.path.join(dir_path, dir), fold_n)
        folds.append(messages)
    return folds


def _read_part(dir_path, fold_n):
    messages = []
    for file_path in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_path)) as file_obj:
            subject, body = _read_message(file_obj)
            is_spam = False if file_path.find(SPAM_LABEL) < 0 else True
            messages.append(Mail(subject, body, file_path, fold_n, is_spam))
    return messages


def _read_message(file_obj):
    all_text = file_obj.read().split('\n')
    assert len(all_text) == 4
    subject, _, body, _ = all_text

    subject = subject.strip().split(' ')[1:]
    body = body.strip().split(' ')

    return subject, body
