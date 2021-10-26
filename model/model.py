from torch import nn
from transformers import BertForTokenClassification

class NERClassifier(nn.module):
    def __init__(self, n_labels, checkpoint, prob):
        super(NERClassifier, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(checkpoint, num_labels=n_labels)

    def forward(self, ids, mask, labels):
        return self.bert(ids, mask, labels = labels)