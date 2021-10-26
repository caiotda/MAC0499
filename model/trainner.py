from preprocess_dataset import NERDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

import torch
import numpy as np

def create_data_loader(df, max_len, batch_size, num_workers):
    ds = NERDataset(df, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class Trainner:

    def __init__(self, device, dataLoader, model, optimizer):
        self.model = model.to(device)
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.epochs = epochs
        self.total_steps = len(dataLoader) * epochs

    def _train_epoch(self):
        model = self.model.train()

        losses = []
        correct = 0

        for sample in self.dataLoader:
            input_tensor = sample["input_ids"]
            att_mask = sample["attention_mask"]
            target = sample["targets"]

            loss, output = model(ids=input, mask=att_mask, labels=target)
            losses.append(loss)
        
        return np.mean(losses)

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


if __name__ == "__main__":
    main()