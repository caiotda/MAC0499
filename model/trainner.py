from preprocess_dataset import NERDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils import batch_to_labels, batch_f1

import torch
import numpy as np

def create_data_loader(df, max_len, batch_size, num_workers):
    ds = NERDataset(df, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class Trainner:

    # TODO: batch_size como parametro do construtor?
    # Uma alternativa é passar um dicionario com alguns parametros
    # de treino também.
    def __init__(self, device, dataLoader, model, optimizer, num_examples, max_len, num_epochs, evaluator):
        self.model = model.to(device)
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.dev = device
        self.batch_len = max_len
        self.num_examples = num_examples
        self.epochs = num_epochs
        self.evaluate = evaluator.evaluate
        #self.total_steps = len(dataLoader) * epochs

    def _train_epoch(self):
        print(f"Treinando em {self.dev}")
        self.model.train() # set model for training mode

        losses = []
        f1_l = []
        temp_f1 = []
        correct_predictions = 0
        temp_accuracy = 0

        relevant_data = temp_relevant_data = 0
        debug_ammount = 100
        for idx, sample in enumerate(self.dataLoader):

            input_tensor = sample["input_ids"].squeeze().to(self.dev, dtype = torch.long)
            att_mask = sample["attention_mask"].to(self.dev, dtype = torch.long)
            target = sample["targets"].to(self.dev, dtype = torch.long)

            out = self.model(input_tensor, att_mask, labels=target)
            logits = out['logits']
            loss = out['loss']
            f1 = batch_f1(true=target, logits=logits)

            # Creates boolean vector for every relevant value
            #active_accuracy = target.view(-1) != -100
            #flat_targets = target.view(-1)
            #flat_logits = logits.view(-1)

            ## Selects the relevant labels
            #relevant_labels = torch.masked_select(flat_targets, active_accuracy)
            #relevant_logits = torch.masked_select(flat_logits, active_accuracy)
            

            #relevant_data += len(relevant_logits)
            #temp_relevant_data += len(relevant_logits)

            #preds = batch_to_labels(relevant_logits)
            #preds = preds.to(self.dev)

            #correct_predictions += torch.sum(preds == relevant_labels).item()
            #temp_accuracy += correct_predictions

            losses.append(loss.item())
            f1_l.append(f1)
            temp_f1.append(f1)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
               parameters=self.model.parameters(), max_norm=10
            )


            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # We print the results every 100 mini batches
            if (idx % debug_ammount == 0):
                prog = "{:.2f}".format(100*idx/len(self.dataLoader))
                print(f"Iteração {idx} -------- Loss: {loss} f1 nas ultimas {debug_ammount} iterações: {np.sum(temp_f1)/debug_ammount} ------ Progresso: {prog}%.")
                temp_f1 = []
        return losses, f1_l

    def train(self):
        print("Começando treino! Essa função retorna a media de f1 e loss em cada epoch de treino e avaliação")
        losses_t_total = []
        losses_e_total = []
        f1_t_total = []
        f1_e_total = []
        for idx in range(self.epochs):
            print(f"----------Começando treino da epoch nº {idx+1}")
            losses_t, f1_t = self._train_epoch() # Eventualmente vou iterar pelas epochs
            losses_e, f1_e = self.evaluate()
            print(f"-------Fim da epoch nº {idx+1}")
            print(f"Dados de treino: Loss media da epoch: {np.mean(losses_t)}; f1 medio da epoch: {np.mean(f1_t)}")
            print(f"Dados de avaliação: Loss media da epoch: {np.mean(losses_e)}; f1 medio da epoch: {np.mean(f1_e)}")
            losses_t_total.append(np.mean(losses_t))
            losses_e_total.append(np.mean(losses_e))
            f1_e_total.append(np.mean(f1_e))
            f1_t_total.append(np.mean(f1_t))
        print(f"FIM DO TREINO! f1 media de treino ao fim de {idx+1} epochs: {np.mean(f1_t_total)}")
        return losses_t_total, losses_e_total, f1_e_total, f1_t_total
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