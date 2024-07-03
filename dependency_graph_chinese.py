
import numpy as np
import spacy
import pickle
import jieba
from spacy.tokens import Doc
import zh_core_web_sm

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("zh_core_web_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text_left,aspect,text_right,k,m):
    # https://spacy.io/docs/usage/processing-text
    #读取虚拟方面节点(需要提前提取专门的数据集对应方面列表)
    fname = "./aspect_xuni_covid-ch.txt"
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    aspect_xuni = fin.readlines()
    fin.close()
    aspect_xuni = aspect_xuni[0].split('.')
    # 读取文件信息
    text = text_left + ' ' + aspect + ' ' + text_right
    tokens = nlp(text)
    tokens1 = list(tokens)
    words = text.split()
    matrix1 = np.zeros((len(words), len(words))).astype('float32')
    aspect_linear_left = text_left.split()[-k:]
    aspect_linear_right = text_right.split()[:k]
    c = 1
    aspect_linear_left_left = str(aspect_linear_left).split()[-c:]
    aspect_linear_right_right = str(aspect_linear_right).split()[:c]
    assert len(words) == len(list(tokens))
    # 句法依存树，非方面词之间的异构规则连接
    for token in tokens:
        matrix1[token.i][token.i] = 1  # 句法依存树 每个词跟自身的连接为1
        for child in token.children:  # 返回依赖token的其他token
            matrix1[token.i][child.i] = 1
            matrix1[child.i][token.i] = 1
    # 方面节点与单词节点之间的异构规则边连接
    for token in tokens1:
        token_str = str(token)
        if token_str == aspect:
            r = token.i
            for attention in words:
                if attention in aspect_linear_right:
                    d = words.index(attention)
                    matrix1[r][d] = 1 - (d + m - r) / k
                    matrix1[d][r] = 1 - (d + m - r) / k
            for attention in words:
                if attention in aspect_linear_left:
                    d = words.index(attention)
                    matrix1[r][d] = 1 - (d + m - r) / k
                    matrix1[d][r] = 1 - (d + m - r) / k
            for attention in words:
                if attention in aspect_linear_right_right:
                    d = words.index(attention)
                    matrix1[r][d] = 1 - (r - d) / k
                    matrix1[d][r] = 1 - (r - d) / k
            for attention in words:
                if attention in aspect_linear_left_left:
                    d = words.index(attention)
                    matrix1[r][d] = 1 - (r - d) / k
                    matrix1[d][r] = 1 - (r - d) / k
            # 方面节点与其他的方面节点之间的虚拟连接
            #for attention in words:
                #if attention in aspect_xuni:
                    #matrix1[token.i][words.index(attention)] = 1
                    #matrix1[words.index(attention)][token.i] = 1
    return matrix1

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    sum=0
    for i in range(0, len(lines), 3):
        lenth=len(lines[i])
        sum += lenth
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]#以方面词分割左右句
        aspect = lines[i + 1].lower().strip()
        k = int(sum/len(lines)/2)
        #k = int(len(lines[i])/7)
        m=2
        adj_matrix = dependency_adj_matrix(text_left,aspect,text_right,k,m)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':
    process('./datasets/covid-2019/covid-2019_train.raw')
    process('./datasets/covid-2019/covid-2019_test.raw')



