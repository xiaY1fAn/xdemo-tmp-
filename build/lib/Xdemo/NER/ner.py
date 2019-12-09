import os, sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import argparse
import logging
from NERTagger import CNNERTagger


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-s', type=str, default=None)
parser.add_argument('-f', type=str, default=None)
parser.add_argument('--output', type=str, default='result.txt')
parser.add_argument('--delimiter', type=str, default=': ')
parser.add_argument('--model_dir', type=str, default='../model-ner/')

if __name__ == "__main__":
    args = parser.parse_args()
    is_sent = args.s
    is_file = args.f
    model_dir = args.model_dir
    delimiter = args.delimiter
    if is_sent and is_file:
        logging.error("The option '-s' and '-f' can't be active in the same time.")
        exit(1)
    if (is_sent is None) and (is_file is None):
        logging.error("The option '-s' and '-f' should choose one.")
        exit(1)

    if is_sent is not None:
        sents = [is_sent]
        NERTagger = CNNERTagger(model_dir)
        ret = NERTagger.label(sents)
        for sent in ret:
            if len(sent) == 0:
                print("None entity.")
            for (e, t) in sent:
                print(delimiter.join([e, t]))

    else:
        Tokenizer = CNNERTagger(model_dir)
        output = args.output
        Tokenizer.predict_file(is_file, output_file=output, conll_format=False, delimiter=delimiter)



