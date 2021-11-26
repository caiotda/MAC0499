import torch
import transformers

from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    """
    implements a pytorch Dataset. As such, it must implement
    3 methods: __len__, __getitem__ and _init__.
    """
    def __init__(self, data, max_len, tokenizer):
        """
        Receives a Dataset object, a Transformers tokenizer
        and the max_len that each entry must have.
        """
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
    
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
        encoded_input = self._retokenize(input_tokens)
        attention_mask = encoded_input['attention_mask']

        input_ids = encoded_input['input_ids']
        # In order to enforce all batch entries with same shape,
        # we artifically pad the labels to be of same length,
        # regardless of the input length.
        labels.extend([0] * self.max_len)
        labels = labels[:self.max_len]
        targets = torch.tensor(labels, dtype=torch.long)
        
        try:
            assert len(input_ids.flatten()) == len(attention_mask.flatten()) == len(targets.flatten())
        except:
            print('Size input_ids: ', len(input_ids.flatten()))
            print('Size attention_mask: ', len(attention_mask.flatten()))
            print('Size targets: ', len(targets.flatten()))
            raise Exception("Length Mismatch in Dataset object.Check input_ids,\
                attention_mask or targets length.") 
        return {
            "id": idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets
        }

    
    def _retokenize(self, entry):
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
        return self.tokenizer(sentence,
                                    max_length=self.max_len,
                                    truncation=True,
                                    padding='max_length', # Trocar isso pelo mais longo do batch. Faz mais sentido pra treino em GPU!
                                    add_special_tokens=False,
                                    return_attention_mask=True,
                                    return_tensors='pt')


def main():
    CHECKPOINT = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)
    data = "lener_br"
    dataset = load_dataset(data)
    ds = NERDataset(dataset['train'], max_len=180, tokenizer=tokenizer)

    print(ds[0])

if __name__ == "__main__":
    main()
