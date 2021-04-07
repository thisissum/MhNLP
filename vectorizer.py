from abc import abstractmethod

import numpy as np
from typing import *

import gensim

class Vectorizer(object):

    def __init__(self, word_tokenizer, word2id=None, embedding_matrix=None):
        self._word_tokenizer = word_tokenizer
        self._word2id = word2id
        self._embedding_matrix = embedding_matrix
        self._emb_dim = embedding_matrix.shape[1]
    
    def __getitem__(self, text: str) -> np.ndarray:
        pass

    def vectorize(self, texts: Iterable[str]) -> np.ndarray:
        pass

    def eval_sim(self, text1: str, text2: str) -> float:
        pass

    def fit(self, texts: Iterable[str], **kwargs):
        pass
    
    def query(self, q: str, keys: Iterable[str], k: int = 1) -> List[str]:
        pass

    def save(self, path):
        pass
    
    @classmethod
    def load(self, path):
        pass


class SentenceVectorizer(Vectorizer):

    def __init__(self, word_tokenizer, word2id=None, embedding_matrix=None):
        super(SentenceVectorizer, self).__init__(word_tokenizer, word2id, embedding_matrix)
        self._embedding = WordVectorizer(word_tokenizer, word2id, embedding_matrix)

    def __getitem__(self, sent: str) -> np.ndarray:
        pass

    def vectorize(self, sents: Iterable[str]) -> np.ndarray:
        pass

    def eval_sim(self, sent1: str, sent2: str) -> float:
        pass

    def fit(self, sents: Iterable[str], **kwargs):
        pass
    
    def query(self, q: str, keys: Iterable[str], k: int = 1) -> List[str]:
        pass




class WordVectorizer(Vectorizer):

    def __getitem__(self, word: str) -> np.ndarray:
        word_idx = self._word2id.get(word, 0)
        return self._embedding_matrix[word_idx]

    def vectorize(self, words: Iterable[str]) -> np.ndarray:
        word_idx_list = [self._word2id.get(word, 0) for word in words]
        seleted = self._embedding_matrix[word_idx_list]
        return seleted

    def eval_sim(self, word1: str, word2: str) -> float:
        word1_vec, word2_vec = self[word1], self[word2]
        sim = np.sum(word1_vec * word2_vec) / (np.sum(word1_vec**2)**0.5 * np.sum(word2_vec**2)**0.5)
        return sim

    def fit(self, sents: Iterable[str], **wvargs):
        pass

    def query(self, q: str, keys: Iterable[str], k:int = 1) -> List[str]:
        pass
