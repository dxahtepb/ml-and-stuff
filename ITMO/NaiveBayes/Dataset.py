import os
import re


class Message:
    words = {}
    headerWords = {}
    bodyWords = {}

    def __init__(self):
        self.words = {}
        
    def add(self, word, type):
        if word in self.words:
            self.words[word] += 1
        else:
            self.words[word] = 1
        if type == 0:
            if word in self.headerWords:
                self.headerWords[word] += 1
            else:
                self.headerWords[word] = 1
        else:
            if word in self.bodyWords:
                self.bodyWords[word] += 1
            else:
                self.bodyWords[word] = 1


class Dataset:
    Ham = []
    Spam = []
    def __init__(self, fileDir):
        fold = -1
        f_name = re.compile("legit")
        for dirs in os.listdir(fileDir):
            fold += 1
            self.Ham.append([])
            self.Spam.append([])
            for file in os.listdir(os.path.join(fileDir, dirs)):
                filename = os.fsdecode(file)
                m_type = []
                if f_name.findall(filename):
                    self.Ham[fold].append(Message())
                    m_type = self.Ham[fold]
                else:
                    self.Spam[fold].append(Message())
                    m_type = self.Spam[fold]
                with open(os.path.join(fileDir, dirs, filename), "r") as inp_file:
                    for ind, line in enumerate(inp_file.readlines()):
                        splitted_line = line.split(" ")
                        for elem in splitted_line:
                            if elem.isdigit():
                                m_type[len(m_type)-1].add(elem, ind)

