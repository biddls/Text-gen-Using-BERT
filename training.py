from glob import glob
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.optim import AdamW
from tqdm import tqdm
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Text_DataSet(torch.utils.data.Dataset):
    def __init__(self, files):
        # load data
        files = glob(files)
        corpus = []
        for file in files:
            with open(file, 'r') as f:
                corpus.append(f.read())

        inputs = tokenizer(
            corpus,
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

def train(epochs, dataset, log=True, _model=None, _device=None):
    if _model is None:
        PATH = "models"
        model = BertForMaskedLM.from_pretrained(PATH, local_files_only=True)
        # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    else:
        model = _model

    if _device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = _device

    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)

    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        if log:
            loop = tqdm(loader, leave=True)
        else:
            loop = loader
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}/{epochs-1}')
            loop.set_postfix(loss=loss.item())
    return model

if __name__ == "__main__":
    if not os.path.exists("bbc"):
        #download the dataset
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen

        # open url
        resp = urlopen("http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip")
        # read zipfile
        zipfile = ZipFile(BytesIO(resp.read()))
        with zipfile as zip_ref:
            zip_ref.extractall("")
        os.remove("bbc/README.TXT")

    if not os.path.exists("onion/data"):
        raise Exception("No onion dataset found")

    model = train(3, Text_DataSet('bbc/**/*.txt'))
    model = train(3, Text_DataSet('onion/data/*.txt'), _model=model)
    model.save_pretrained('models/')
