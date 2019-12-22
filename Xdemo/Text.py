import re
import logging
import copy
import math
import pypinyin

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sentence_cut(sents):
    """Cut sentences.
       Args:
           sents: A string, some sentences.
    """
    if not isinstance(sents, str):
        logging.error("Input should be str type.")
        exit(-1)

    punct = r'([.?!。？！])'
    sentences = re.split(punct, sents)
    sentences.append('')
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]

    if len(sentences) > 1 and sentences[-1] == '':
        sentences.pop()
    return sentences


def count(sents, w):
    """Count the times of w appear in sents.
       Args:
           sents: A string of a list of string.
           w: The count word.
    """
    if isinstance(sents, str):
        return sents.count(w)
    elif isinstance(sents, list):
        n = 0
        for st in sents:
            n += st.count(w)
        return n
    else:
        logging.error('The first input should be a string or a list of string.')
        exit(-1)


def similarity(text1, text2):
    """Calculate the similarity of two text.
       1 - (Min edit distance / max length) for now.
    """
    n1 = len(text1)
    n2 = len(text2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dist = [[0 for i in range(n2+1)] for j in range(n1+1)]
    for i in range(n2+1):
        dist[0][i] = i
    for i in range(n1+1):
        dist[i][0] = i
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            dist[i][j] = min(dist[i-1][j], dist[i][j-1], dist[i-1][j-1]) + 1
            if text1[i-1] == text2[j-1] and dist[i][j] > dist[i-1][j-1]:
                dist[i][j] = dist[i-1][j-1]
    return 1 - dist[n1][n2] / max(n1, n2)




class TextCollection():
    """Text collection."""
    def __init__(self, text=None):
        """Initial the text collection.
           Args:
               text: A list of text(str).
        """
        self._texts = copy.deepcopy(text) if text is not None else ['']
        self._idf_cache = {}

    def tf(self, term, text):
        """The frequency of the term in text."""
        return text.count(term) / len(text)

    def idf(self, term):
        """idf"""
        idf = self._idf_cache.get(term)
        if idf is None:
            matches = len([True for text in self._texts if term in text])
            idf = (math.log(len(self._texts) / matches) if matches else 0.0)
            self._idf_cache[term] = idf
        return idf

    def tf_idf(self, term, text):
        """tf*idf"""
        return self.tf(term, text) * self.idf(term)


def match(texts, substr):
    """Text match.
       Ctrl/command + F for now.
    """
    ret = []
    for t in texts:
        if substr in t:
            ret.append(t)
    return ret


def get_pinyin(text, tone='normal', duoyin=False):
    if tone == 'normal':
        text_pinyin = pypinyin.pinyin(text, style=pypinyin.NORMAL, heteronym=duoyin)
    else:
        text_pinyin = pypinyin.pinyin(text, heteronym=duoyin)
    if duoyin:
        return text_pinyin
    else:
        ret = ''
        for w in text_pinyin:
            ret += ''.join(w) + ' '
        return ret

class Phrase(object):
    '''短语提取'''
    def __init__(self, text=None):
        '''
        init
        :param text: [sent, sent, ...]
        '''
        self._texts = copy.deepcopy(text) if text is not None else ['']

    def get_phrase(self, min_frequency=2, min_length=2):
        '''

        :return: a phrase dict: {ph1: n, ph2: n, ...}
        '''
        tmp_dic = {}
        for sent in self._texts:
            for i in range(len(sent) - min_length + 1):
                tmp_p = sent[i:i+min_length]
                if tmp_p in tmp_dic:
                    tmp_dic[tmp_p] += 1
                else:
                    tmp_dic[tmp_p] = 1
        to_delete_list = []
        for ph in tmp_dic:
            if tmp_dic[ph] < min_frequency:
                to_delete_list.append(ph)
        for ph in to_delete_list:
            tmp_dic.pop(ph)
        return tmp_dic

    def get_most_frequent_phrase(self, min_frequency=1, min_length=1, num=1):
        tmp_dic = {}
        for sent in self._texts:
            for i in range(len(sent) - min_length + 1):
                tmp_p = sent[i:i + min_length]
                if tmp_p in tmp_dic:
                    tmp_dic[tmp_p] += 1
                else:
                    tmp_dic[tmp_p] = 1
        to_delete_list = []
        for ph in tmp_dic:
            if tmp_dic[ph] < min_frequency:
                to_delete_list.append(ph)
        for ph in to_delete_list:
            tmp_dic.pop(ph)
        ret = [(k, v) for (k, v) in tmp_dic.items()]
        ret.sort(key=lambda x:x[1], reverse=True)
        ret = ret[:num]
        return ret






if __name__ == '__main__':
    '''
    ret = sentence_cut('haha.uuu?enen!hehe。我是，一个狼人！信吗？哈哈。和')
    print(ret)
    ret = sentence_cut('')
    print(ret)
    '''
    t1 = 'tyii'
    t2 = 'tyui'
    print(get_pinyin("行", tone='symbol', duoyin=True))
    tt = ['我们是中国人', '我们是共产主义接班人']
    ff = Phrase(tt)
    print(ff.get_most_frequent_phrase(min_length=4, num=2))
    print(ff.get_phrase(min_length=2))