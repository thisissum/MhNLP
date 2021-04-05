import numpy as np
from typing import *

class Vectorizer(object):
    pass


class SentenceVectorizer(Vectorizer):
    def __getitem__(self, sent: str) -> np.ndarray:
        pass

    def vectorize(self, sent: str) -> np.ndarray:
        pass

    def eval_sim(self, sent1: str, sent2: str) -> float:
        pass

    def fit(self, sents: Iterable[str]):
        pass

    



class OneHotEmbedding(Vectorizer):
    pass


class WordEmbedding(Vectorizer):
    pass


class BertEmbedding(Vectorizer):
    pass