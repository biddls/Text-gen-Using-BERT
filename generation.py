from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Text_DataSet(torch.utils.data.Dataset):
    def __init__(self, text: int):

        inputs = tokenizer(
            [text],
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
        )

        inputs['labels'] = inputs.input_ids.detach().clone()

        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)

        # create mask array
        # doesn't mask over any PAD, CLS or SEP tokens (0, 101, 102)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
                   (inputs.input_ids != 102) * (inputs.input_ids != 0)

        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )

        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103

        self.encodings = inputs
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def gen(text: dict):
    PATH = "models"
    model = BertForMaskedLM.from_pretrained(PATH, local_files_only=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # and move our model over to the selected device
    model.to(device)

    # activate inferance mode
    model.eval()

    # pull all data required for generation
    input_ids = text['input_ids'].to(device)
    attention_mask = text['attention_mask'].to(device)
    labels = text['labels'].to(device)

    # process
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    labels=labels)
    return outputs

if __name__ == "__main__":
    pass
