import pandas as pd
from torch.utils.data.dataset import Dataset
import csv

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import torch
class CustomDataset(Dataset):

    def __init__(self, dict_path, dataframe,  max_length_sentences=30, max_length_word=35):
        self.data = dataframe
        self.text = dataframe.Text
        self.labels = self.data.list
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = 34

        #self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        label = self.labels[index]
        text = str(self.text[index])
        #print(text)
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        #label=torch.FloatTensor(label)


        return document_encode.astype(np.int64), label
if __name__ == '__main__':
    with open("../dl_train_feature","rb") as f:
        df=pickle.load(f)
    test = CustomDataset(dict_path="glove.6B.50d.txt",dataframe=df,max_length_sentences=130, max_length_word=235)
    print (test.__getitem__(index=2)[0].shape)