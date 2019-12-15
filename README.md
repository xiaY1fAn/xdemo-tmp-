# NLP工具

## 当前需要实现

 功能|完成|测试   
 :-:|:-:|:-:  
 分句|Y|  
 词频统计|Y|  
 tf-idf|Y|  
 文本相似度|Y|  
 词典|Y|  
 文本匹配|Y|  
 自动注音|1|  
 短语提取|1|  
 简繁转换|1|  
 Embedding|1|  
 分词|Y|  
 POS|Y|  
 NER|Y|  
 句法分析|2|  
 语义角色标注|2|    
 语言模型|2|  
 网页展示优化|3|  
 
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


