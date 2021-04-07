import numpy as np
from typing import *

import gensim

class Vectorizer(object):

    def save(self, path):
        pass
    
    @classmethod
    def load(self, path):
        pass


class SentenceVectorizer(Vectorizer):

    def __init__(self, embedding):
        pass

    def __getitem__(self, sent: str) -> np.ndarray:
        pass

    def vectorize(self, sents: Iterable[str]) -> np.ndarray:
        pass

    def eval_sim(self, sent1: str, sent2: str) -> float:
        pass

    def fit(self, sents: Iterable[str], method: str = "tfidf", **kwargs):
        pass
    
    def query(self, q: str, keys: Iterable[str], k: int = 1) -> List[str]:
        pass




class WordVectorizer(Vectorizer):

    def __init__(self, word_tokenizer, word2id = None, embedding_matrix = None):
        pass

    def __getitem__(self, word: str) -> np.ndarray:
        pass

    def vectorize(self, words: Iterable[str]) -> np.ndarray:
        pass

    def eval_sim(self, word1: str, word2: str) -> float:
        pass

    def fit(self, sents: Iterable[str], **wvargs):
        pass

    def query(self, q: str, keys: Iterable[str], k:int = 1) -> List[str]:
        pass
