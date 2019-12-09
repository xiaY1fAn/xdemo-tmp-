
import os, sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import numpy as np
import logging
import codecs
import torch
from model.pytorch_pretrained_bert.tokenization  import BertTokenizer
from model.SequenceLabeling_bert import BertForSeqLabeling
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

entity_set = {'PER': '人物', 'LOC': '地点', 'MOV': '影视作品', 'NOV': '网络小说', 'WEB': '网站', \
                  'NAT': '国家', 'SCH': '学校', 'COM': '企业', 'SON': '歌曲', 'MUS': '音乐专辑', \
                  'VIE': '景点', 'CIT': '城市', 'TVS': '电视综艺'}


class InputFeatures(object):

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


class CNNERTagger():

    def __init__(self, model_dir, batch_size=8, no_cuda=False):
        if not os.path.exists(model_dir):
            logging.error("%s is not a valid model path." % model_dir)
            exit(1)
        if len(set(['bert_config.json', 'checkpoint', 'vocab.txt']) - set(os.listdir(model_dir))) > 0:
            logging.error('%s not a valid model directory', model_dir)
            exit(1)
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint'), map_location='cpu')
        self._max_seq_length = checkpoint['max_seq_length']
        self._label_list = checkpoint['label_list']
        logger.info("Loading the model...")
        self._tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=checkpoint['lower_case'])
        self._model = BertForSeqLabeling.from_pretrained(model_dir, state_dict=checkpoint['model_state'],
                                                         num_labels=len(self._label_list))
        self._batch_size = batch_size
        if not no_cuda and torch.cuda.is_available():
            self._device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        logger.info("Model is ready...")

    def _split_paragraph(self, ori_lines):
        lines = []
        for index, line in enumerate(ori_lines):
            if len(line) > self._max_seq_length - 2:
                last_index = 0
                for s_index, c in enumerate(line):
                    if c == '。' or c == '.':
                        lines.append(line[last_index:s_index + 1])
                        last_index = s_index + 1
                if last_index < len(line):
                    lines.append(line[last_index:])
            else:
                lines.append(line)
        return lines

    def _get_features(self, ori_lines):
        lines = self._split_paragraph(ori_lines)
        features = []
        tokens_list = []
        for index, line in enumerate(lines):
            tokens = ['[CLS]']
            chars = self._tokenizer.tokenize(line)
            tokens.extend(chars)
            if len(tokens) > self._max_seq_length - 1:
                logging.debug('Example {} is too long: {}'.format(index, line))
                tokens = tokens[0:(self._max_seq_length - 1)]
            tokens.append('[SEP]')
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            zero_padding = [0] * (self._max_seq_length - len(input_ids))
            input_ids += zero_padding
            input_mask += zero_padding

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask))
            tokens_list.append(tokens[1:-1])
        return features, tokens_list

    def _predict_features(self, features, tokens):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

        predict_data = TensorDataset(all_input_ids, all_input_mask)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=self._batch_size)
        self._model.eval()
        predict_ids = []
        for batch in predict_dataloader:
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask = batch
            logits = self._model(input_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            predict_ids.extend(np.argmax(logits, -1).tolist())
        predictions = []
        for token_line, predict_line in zip(tokens, predict_ids):
            predictions.append([self._label_list[label_id] for label_id in predict_line[1: 1+len(token_line)]])
        return predictions

    def label(self, sentences):
        features, tokens = self._get_features(sentences)
        predictions = self._predict_features(features, tokens)
        ret = []
        for sen_predictions, sen_tokens in zip(predictions, tokens):
            ret.append([(tag, token) for tag, token in zip(sen_predictions, sen_tokens)])
        ret = self.get_entity(ret)
        return ret

    def get_entity(self, batched_labeled_list):
        entities = []
        for sent in batched_labeled_list:
            entity = []
            word = ''
            type = ''
            for token in sent:
                l = token[0].split('-')
                w = token[1]
                if l[0] == 'B':
                    if word != '':
                        entity.append((word, entity_set[type]))
                    type = l[1]
                    word = w
                elif l[0] == 'I':
                    word += w
                else:
                    if word != '':
                        entity.append((word, entity_set[type]))
                        word = ''
            if word != '':
                entity.append((word, entity_set[type]))
            entities.append(entity)
        return entities


    def predict_file(self, input_file_path, output_file, conll_format, delimiter):
        if not conll_format:
            sentences = []
            for line in codecs.open(input_file_path, 'r', encoding='utf-8'):
                sentences.append(line.strip('\n'))
            features, tokens = self._get_features(sentences)
        else:
            sentences = []
            words = []
            for line in codecs.open(input_file_path, 'r', encoding='utf-8'):
                if not line.strip():
                    sentences.append(words)
                    words = []
                else:
                    segs = line.split()
                    words.append(segs[0])
            features, tokens = [], []
            for index, line in enumerate(sentences):
                line_tokens = ['[CLS]']
                for w in line:
                    chars = self._tokenizer.tokenize(w)
                    if not chars:
                        chars = ['[UNK]']
                    line_tokens.extend(chars)
                if len(line_tokens) > self._max_seq_length - 1:
                    logging.debug('Example {} is too long: {}'.format(index, line))
                    line_tokens = line_tokens[0:(self._max_seq_length - 1)]
                line_tokens.append('[SEP]')
                input_ids = self._tokenizer.convert_tokens_to_ids(line_tokens)
                input_mask = [1] * len(input_ids)
                zero_padding = [0] * (self._max_seq_length - len(input_ids))
                input_ids += zero_padding
                input_mask += zero_padding
                features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask))
                tokens.append(line_tokens[1:-1])
        predictions = self._predict_features(features, tokens)
        tokens_writer = codecs.open(output_file, 'w', encoding='utf-8')

        ret = []
        for sen_predictions, sen_tokens in zip(predictions, tokens):
            ret.append([(tag, token) for tag, token in zip(sen_predictions, sen_tokens)])
        token_sents = self.get_entity(ret)

        for sent in token_sents:
            if len(sent) == 0:
                tokens_writer.write('None\n')
            else:
                r = ''
                for (e, t) in sent:
                    r += delimiter.join([e, t]) + '  '
                tokens_writer.write(r + '\n')

        tokens_writer.close()


if __name__ == "__main__":
    model = CNNERTagger("../model-ner/")
    print(model.label(["《陈情令》真好看！"]))
    pass


