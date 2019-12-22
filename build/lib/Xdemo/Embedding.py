import numpy as np

class Embd(object):
    '''Embedding'''
    def __init__(self, file, binaray=False, dim=None, delimiter=None):
        """
        init
        :param file: Pretrained embedding file
        """
        if binaray:
            f = open(file, 'rb')
        else:
            f = open(file, 'r')

        self._vector = {}
        len_dim = 0
        for line in f:
            line = line.strip().split(delimiter)
            word = line[0]
            vec = line[1:]
            vec = [float(e) for e in vec]
            if len_dim == 0:
                len_dim = len(vec)
            else:
                if len_dim != len(vec):
                    raise Exception("Embedding dim is not the same.")
            self._vector[word] = vec
        f.close()
        self.dim = dim if dim is not None else len_dim

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        return self._vector[item] if item in self._vector else None

    def similar(self, word, num=5):
        """
        Find similar words.
        :param word:
        :return: The similar words as word.
        """
        if word not in self._vector:
            print(word + ' is not in Embedding.')
            return None
        v_word = self._vector[word]
        v_word = np.array(v_word)
        f1 = np.linalg.norm(v_word)
        tmp = []
        for w in self._vector:
            if w == word:
                continue
            v_w = np.array(self._vector[w])
            dist = np.dot(v_word, v_w) / (f1 * (np.linalg.norm(v_w)))
            tmp.append((w, dist))
        tmp.sort(key=lambda x: x[1], reverse=True)
        tmp = tmp[:num]
        tmp = [m[0] for m in tmp]
        return tmp


if __name__ == '__main__':
    Em = Embd('a.txt')
    print(Em.similar('b', num=1))
    print(Em['c'])