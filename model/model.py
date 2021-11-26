from torch import nn
from transformers import BertForTokenClassification

import torch

class NERClassifier(nn.Module):
    def __init__(self, n_labels, checkpoint, prob=0.3):
        super(NERClassifier, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(checkpoint, num_labels=n_labels)

    def forward(self, ids, mask, labels=None):
        return self.bert(ids, mask, labels=labels)

def main():
    t = [  192,  7463,  8427, 22301,   131, 12127,  9008, 22301, 22402, 16484,
          187, 22360, 22339,  9008,   118,   177, 22402, 16484, 10836, 13760,
         7545, 22320, 22323, 22351, 22301, 22402, 16484,   212,  8718,   250,
         7665,  6072,   213,  8718, 22301,  6538,   118, 11635,  9008, 13270,
         7073,  6765,   118, 11741, 22328, 22341,  6392, 22301,   212,  9008,
        22317,   213,  7073,  6538, 22321, 22352, 21748, 22317,   212, 22371,
        22318, 22327,  6162, 22317,   192, 22311,   278,  5650, 22341,   257,
         5476, 15289,  5903, 22327,   118,   248, 18199,  6392, 11836, 22309,
          118,   177, 10409, 22420, 22320, 14298, 22301, 10836, 13760, 16017,
        22322, 22339, 12547, 22402, 16484, 15040, 18868, 22322, 22349, 22341,
         9208,   248, 22301, 13760, 11846, 22379, 22320, 14298, 22301,   177,
         5226, 22341, 22317,   118, 11635,  3341, 12547, 22402, 22301, 10836,
        13760,  4529,  5869, 22351,   118, 11635, 22309, 22333, 22341, 22360,
        22351, 22317,   192, 22348,  6538, 16017,  8427, 22309,   118, 11635,
         9008, 13270,  7073,  6765, 11247,  7918, 22340,  6392, 22301,   118,
          248, 18199,  6392, 11836, 22309,   257,  5476, 12234, 22340,  5476,
         6392, 22301,   119,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0]

    labels = torch.tensor([i for i in range(0,12)])
    mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = torch.tensor([mask])
    m = NERClassifier(n_labels=12, checkpoint='neuralmind/bert-base-portuguese-cased')


    print(labels.size())
    a = m(torch.tensor([t]), mask=mask)
    print(a)
if __name__ == '__main__':
    main()
