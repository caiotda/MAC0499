import torch
import transformers

from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

CHECKPOINT = 'neuralmind/bert-base-portuguese-cased'
TOKENIZER = BertTokenizer.from_pretrained(CHECKPOINT)


def extend_labels(labels):
    return labels

def retokenize(entry):
    """ []string -> []string
    GIVEN an array of strings
    RETURNS an array of strings

    Receives an array of strings that represents
    a sentence that was tokenized using whitespaces.
    The function then joins the words into a sentence, then
    performs subword tokenization using BERTimbau's tokenizer.

    This is done to avoid occurence of OOV tokens.
    """

    sentence = " ".join(entry)
    print(f"Comprimento da sentença de entrada: {len(sentence)}")
    return TOKENIZER.encode_plus(sentence,
                                max_length=256,
                                padding='max_length', # Trocar isso pelo mais longo do batch. Faz mais sentido pra treino em GPU!
                                add_special_tokens=True,
                                return_attention_mask=True,
                                return_tensors='pt')



class NERDataset(Dataset):
    """
    implements a pytorch Dataset. As such, it must implement
    3 methods: __len__, __getitem__ and _init__.
    """

    def __init__(self, data, tokenizer, max_len):
        """
        Receives a Dataset object, a Transformers tokenizer
        and the max_len that each entry must have.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        """
        Dataset class already implements a __len__
        method, so we just refer to it.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Given an index, retrieves an entry from self.data and retokenizes it.
        Because we're tokenizing using subword tokenization, this process will
        generally expand the size of the sentence. Therefore, we also expand
        the label to each item to reflect this change.
        """

        input_tokens = self.data[idx]['tokens']
        labels = self.data[idx]['ner_tags']
        subword_tokenized = retokenize(input_tokens)

        extended_labels = extend_labels(labels)
        """ try:
            assert len(extended_labels) == len(subword_tokenized['input_ids'])
        except AssertionError:
            print("The extended label should be the same length of retokenized input") """

        print('opa')
        return {
            "id": idx,
            "input_text": input_sentence,
            "input_ids": subword_tokenized['input_ids'].flatten(),
            "attention_mask": subword_tokenized['attention_mask'].flatten(),
            "targets": torch.tensor(extended_labels, dtype=torch.long)
        }


def main():
    data = "lener_br"
    dataset = load_dataset(data)
    teste = NERDataset(dataset['train'])

    print(teste[0])
