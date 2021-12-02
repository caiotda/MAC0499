import torch

import torch
from torch.nn import Sigmoid


def preds_to_label_idx(logits):
    s = Sigmoid()
    probs = s(logits)
    preds = torch.tensor([torch.argmax(prob).item() for prob in probs])

    return preds