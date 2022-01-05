from preprocess_dataset import NERDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils import batch_to_labels, batch_f1

import torch
import numpy as np

class Evaluator:
    def __init__(self, eval_dataset, model, dev):
        self.eval_dataset = eval_dataset
        self.model = model
        self.dev = dev
        
    def evaluate(self):
        """
        Given a refined model, a test dataset and a number of epochs, 
        evaluat
        """

        self.model.eval()


        debug_ammount = 100
        losses = []
        f1_l = []
        temp_f1 = []

        with torch.no_grad():
            for idx, sample in enumerate(self.eval_dataset):

                    input_tensor = sample["input_ids"].squeeze().to(self.dev, dtype = torch.long)
                    att_mask = sample["attention_mask"].to(self.dev, dtype = torch.long)
                    target = sample["targets"].to(self.dev, dtype = torch.long)

                    out = self.model(input_tensor, att_mask, labels=target)
                    logits = out['logits']
                    loss = out['loss']
                    f1 = batch_f1(true=target, logits=logits)

                    losses.append(loss.item())
                    f1_l.append(f1)
                    temp_f1.append(f1)

                    if (idx % debug_ammount == 0 and idx != 0):
                        prog = "{:.2f}".format(100*idx/len(self.eval_dataset))
                        print(f"Iteração {idx} -------- Loss: {loss} f1 nas ultimas {debug_ammount} iterações: {np.sum(temp_f1)/debug_ammount} ------ Progresso: {prog}%.")
                        temp_f1 = []
        del out
        del loss
        del f1
        del logits
        torch.cuda.empty_cache()
        return losses, f1_l