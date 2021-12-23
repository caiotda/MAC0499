import torch

import torch
from torch.nn import Sigmoid
from sklearn.metrics import f1_score as f1

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

def f1_score(true, pred, average='weighted'):
    """
    Calculates the f1_score between two tensors: the true labels tensor,
    and the prediction tensor, outputed by the model. The f1 score is weighted
    by default in order to reduce effect of a imbalanced dataset
    """
    assert type(pred) == torch.Tensor
    assert type(true) == torch.Tensor
    print(pred.shape)
    print(true.shape)
    assert pred.shape == true.shape
    
    return f1(true, pred, average=average)

def batch_f1(true, logits, average='weighted'):
    """
    Given a batch of logits and labels, calculates the f1 score in the batch.
    This is done by concatenating the prediction and label for each example into a
    long array. Then we just call the function f1_score
    """
    pred = batch_to_labels(logits)
    true = true.cpu().view(-1)
    pred = pred.cpu().view(-1)
    
    mask = torch.tensor(list(map(lambda i: True if i != -100 else False, true)))
    
    relevant_preds = torch.masked_select(pred, mask)
    relevant_labels = torch.masked_select(true, mask)
    
    assert relevant_labels.shape == relevant_preds.shape
    
    return f1_score(true=relevant_labels, pred=relevant_preds, average=average) 
    
    
    
    
             
    