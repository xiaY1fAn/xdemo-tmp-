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


