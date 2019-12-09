
import torch

from .pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


class BertForSeqLabeling(PreTrainedBertModel):

    def __init__(self, config, num_labels):
        super(BertForSeqLabeling, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.hidden2label = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask):
        bert_layer, _ = self.bert(input_ids=input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        logits = self.hidden2label(bert_layer)
        return logits
