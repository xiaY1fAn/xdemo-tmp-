# NLP工具
 
 ## 部分功能使用说明  
 ### 分句
 ```python
from Xdemo import Text
ret = Text.sentence_cut('s1s2s3')
ret
['s1', 's2', 's3']
```

### 词频统计
```python
from Xdemo import Text
ret = Text.count(sents, w)
sents可以是句子或句子列表
```
### tf-idf
```python
from Xdemo import Text
cnt = Text.TextCollection(sents)
tfidf = cnt.tf_idf(term, text)
```

### 文本相似度
```python
from Xdemo import Text
similarity = Text.similarity(text1, text2)
暂时使用 1 - （最短编辑距离 / 较长句子长度）作为相似度。  
后续计划使用embedding或者语言模型计算相似度。
```

### 词典
```python
目前只支持生成词典  
python Vocabulary.py file > vocab

```

### 文本匹配
```python
from Xdemo import Text
ret = Text.match(texts, substr)
目前只支持精确匹配，后续计划增加模糊匹配和支持语义层次的匹配算法。

```

### 注音
```python
from Xdemo import  Text
ret = Text.get_pinyin(text)

```

### 短语提取

```python
from Xdemo import Text
ph = Text.Phrase(texts)
ret = ph.get_phrase(min_length=2, min_frequency=2)
ret = ph.get_most_frequent_phrase(min_length=2, min_frequency=2, num=5)

```

### Embedding

```python
from Xdemo import Embedding
Em = Embedding.Embd(file, dim=300, binaray=False) //file是Embedding文件，格式为word dim1 dim2 ...
word_em = Em[word]       //读取Embedding
word_similar = Em.similar(word, num=5) //相似词

```
 
 


## 快速使用
```python
分词
pyhon Tokenize.py -s sentence --model_dir model_dir #句子
pyhon Tokenize.py -f input_file --model_dir model_dir #文件
    optional：
        --output
        --delimiter
        
NER、POS Tagging同上

```

## 作为包使用

```python
安装
pip install dist/xdemo-0.0.1-py3-none-any.whl

分词
from xdemo.Tokenizer.Tokenizer import CNTokenizer
model = CNTokenizer(model_dir)

result = model.label(sents)
#sents 的格式为[sent1, sent2, ...]
#result 的格式为[[s e n t 1], [s e n t 2], ...]

NER、POS Tagging 同上。
```


