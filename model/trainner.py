from preprocess_dataset import NERDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils import batch_to_labels

import torch
import numpy as np

def create_data_loader(df, max_len, batch_size, num_workers):
    ds = NERDataset(df, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class Trainner:

    # TODO: batch_size como parametro do construtor?
    # Uma alternativa é passar um dicionario com alguns parametros
    # de treino também.
    def __init__(self, device, dataLoader, model, optimizer, num_examples, max_len, num_epochs):
        self.model = model.to(device)
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.dev = device
        self.batch_len = max_len
        self.num_examples = num_examples
        self.epochs = num_epochs
        #self.total_steps = len(dataLoader) * epochs

    def _train_epoch(self):
        print(f"Treinando em {self.dev}")
        self.model.train() # set model for training mode

        losses = []

        correct_predictions = 0
        
        for idx, sample in enumerate(self.dataLoader):

            input_tensor = sample["input_ids"].squeeze().to(self.dev, dtype = torch.long)
            att_mask = sample["attention_mask"].to(self.dev, dtype = torch.long)
            target = sample["targets"].to(self.dev, dtype = torch.long)

            out = self.model(input_tensor, att_mask, labels=target)
            logits = out['logits']
            loss = out['loss']
            
            preds = batch_to_labels(logits)
            preds = preds.to(self.dev)

            correct_predictions += torch.sum(preds == target).item()
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if (idx % 100 == 0):
                acc = correct_predictions / (self.num_examples * self.batch_len )
                acc = "{:.4f}".format(acc*100)
                prog = "{:.2f}".format(100*idx/len(self.dataLoader))
                print(f"Iteração {idx} -------- Acuracia: {acc}%------- Loss: {loss} ------ Progresso: {prog}%.")
        self.optimizer.zero_grad()
        return losses, acc

    def train(self):
        loss_total = []
        for idx in range(self.epochs):
            print(f"----------Começando treino da epoch nº {idx+1}")
            losses, acc = self._train_epoch() # Eventualmente vou iterar pelas epochs
            print(f"-------Fim da epoch nº {idx+1}. Loss media da epoch: {np.mean(losses)}")
            loss_total.append(losses)
        print(f"FIM DO TREINO! Loss media ao fim de {idx+1} epochs: {np.mean(loss_total)}")
        return loss_total, acc
def main():

    BATCH_SIZE = 16
    # Informação a ser repassada ao trainer por um cliente
    # de treino, não aqui
    MAX_LEN = 256 
    # Numero qualquer. Essa informação tem nos experimentos que
    # ja rodei

    data = "lener_br"
    df = load_dataset(data)
    train_data_loader = create_data_loader(df['train'], MAX_LEN, BATCH_SIZE, num_workers=4)
    test_data_loader = create_data_loader(df['test'], MAX_LEN, BATCH_SIZE, num_workers=4)
    validation_data_loader = create_data_loader(df['validation'], MAX_LEN, BATCH_SIZE, num_workers=4)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()