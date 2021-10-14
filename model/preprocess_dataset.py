import torch
import transformers

from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

CHECKPOINT = 'neuralmind/bert-base-portuguese-cased'
TOKENIZER = BertTokenizer.from_pretrained(CHECKPOINT)


def extend_labels(labels, new_tokens):
    """
    Given a retokenized input `new_tokens`, extends the
    labels every time the corresponding word was tokenized into
    subwords. this function simply extends the first tag.
    For example:

    ["Calçadão", "de", "Osasco"] <> [B-LOCAL, I-LOCAL, I-LOCAL]
    \/ (Retokenization)
    ["cal", "##ça"", "##dão", "de", "Osa", "##s", "##co"]
    <>
    [B-LOCAL, X, X, I-LOCAL, I-LOCAL, X, X]

    Bert, when performing NER tasks, ignores the tag attributed to a 
    subword, so we ignore subword labels so that metrics are not changed.
    """
    label_idx = 0
    new_labels = []
    relevant_tokens = [token for token in new_tokens if token != '[PAD]']
    for token in relevant_tokens:
        if ("##" in token):
            new_labels.append(-1)
        else:
            new_labels.append(labels[label_idx])
            label_idx += 1
            
    return new_labels

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
    return TOKENIZER.encode_plus(sentence,
                                max_length=256,
                                padding='max_length', # Trocar isso pelo mais longo do batch. Faz mais sentido pra treino em GPU!
                                add_special_tokens=False,
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
        encoded_input = retokenize(input_tokens)

        input_ids = encoded_input['input_ids'].flatten()
        decoded_input = TOKENIZER.convert_ids_to_tokens(input_ids)
        extended_labels = extend_labels(labels, decoded_input)

        return {
            "id": idx,
            "input_text": " ".join(input_tokens),
            "input_ids": input_ids,
            "attention_mask": encoded_input['attention_mask'].flatten(),
            "targets": torch.tensor(extended_labels, dtype=torch.long)
        }


def main():
    data = "lener_br"
    dataset = load_dataset(data)
    teste = NERDataset(dataset['train'], tokenizer=TOKENIZER, max_len=180)

    print(teste[0])

main()