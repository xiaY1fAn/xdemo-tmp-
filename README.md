#NLP工具

##快速使用
```python
分词
pyhon Tokenize.py -s sentence --model_dir model_dir #句子
pyhon Tokenize.py -f input_file --model_dir model_dir #文件
    optional：
        --output
        --delimiter
        
NER、POS Tagging同上

```

##作为包使用

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


