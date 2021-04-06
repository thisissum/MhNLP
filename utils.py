import re
import jieba
from typing import *


def sentence_tokenize(text: str, pattern: str = r"？|。|；|！"):
    return re.split(pattern, text)


def word_tokenize(text: str, package="jieba"):
    if package == "jieba":
        return jieba.cut(text)
    else: 
        raise ValueError("{} not supported.".format(package))


def remove_stopwords(words: List[str], stopwords: Iterable[str]):
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)
    return [word for word in words if word not in stopwords]


def load_stopwords(path="./resource/stopwords.txt"):
    with open(path, mode='r', encoding="utf-8") as f: 
        stopwords = list(map(lambda x: x.replace("\n",""), f.readlines()))
    return stopwords


def drop_url(text: str, pattern: str=r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'):
    return re.sub(pattern, "", text)


def drop_html(text: str, pattern: str=r"<.*?>"):
    output = [piece for piece in re.sub(pattern, " ", text).split(" ") if piece]
    return " ".join(output)

def edit_distance(text1, text2):
    pass



if __name__ == "__main__":
    text = "<tag>aaa</tag><html>bbb</html><cccc>"
    print(drop_html(text))

