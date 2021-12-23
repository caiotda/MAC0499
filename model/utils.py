import torch

import torch
from torch.nn import Sigmoid

def extract_max(sentence):
    return [torch.argmax(word).item() for word in sentence]

def preds_to_label_idx(logits):
    """
    Given an torch.tensor of logits of shape (NUM_WORDS, NUM_LABELS),
    outputs the predictions.
    """
    s = Sigmoid()
    probs = s(logits)
    preds = torch.tensor(extract_max(probs))

    return preds

def batch_to_labels(batch_logits):
    return torch.tensor([preds_to_label_idx(logits).tolist() for logits in batch_logits.squeeze()])