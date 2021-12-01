from preprocess_dataset import NERDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

import torch
import numpy as np

def create_data_loader(df, max_len, batch_size, num_workers):
    ds = NERDataset(df, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class Trainner:

    # TODO: batch_size como parametro do construtor?
    # Uma alternativa é passar um dicionario com alguns parametros
    # de treino também.
    def __init__(self, device, dataLoader, model, optimizer):
        self.model = model.to(device)
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.dev = device
        #self.epochs = epochs
        #self.total_steps = len(dataLoader) * epochs

    def _train_epoch(self):
        self.model.train() # set model for training mode

        losses = []
        correct = 0

        correct_predictions = 0
        for sample in self.dataLoader:
            input_tensor = sample["input_ids"].squeeze().to(self.dev, dtype = torch.long)
            att_mask = sample["attention_mask"].to(self.dev, dtype = torch.long)
            target = sample["targets"].to(self.dev, dtype = torch.long)

            out = self.model(input_tensor, att_mask, labels=target)
            preds = out['logits']
            loss = out['loss']
            #correct_predictions += torch.sum(preds == target) TODO: ainda não posso usar isso! preciso antes converter o tensor pra um vetor de classificações.
            # TODO: tem algo bem parecido no notebook de cliente
            losses.append(loss)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        return losses, np.mean(losses)

    def train(self):
        return self._train_epoch() # Eventualmente vou iterar pelas epochs
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