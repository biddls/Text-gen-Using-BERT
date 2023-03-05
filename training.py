from glob import glob
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import torch
import util


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using: cuda' if torch.cuda.is_available() else 'Using: CPU')


class Text_DataSet(torch.utils.data.Dataset):
    def __init__(self, _files):
        # load data
        files = glob(_files)
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
        print(f'Dataset "{_files}" loaded')
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def train(epochs, train_data, test_data, val_data, _model=None):
    if _model is None:
        model = BertForMaskedLM.from_pretrained("results/checkpoint-500")
    else:
        model = _model

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        do_eval=True, # evaluate the model after training
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=6,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=test_data  # evaluation dataset
    )

    model.to(device)
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"{eval_results['eval_loss']=:.2f}")
    return model

if __name__ == "__main__":
    util.dataset_checks()

    # loads data set
    bbc = Text_DataSet('bbc/**/*.txt')
    # splits data set into train, test and validation sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(bbc, util.split_proportionally(len(bbc), [70, 20, 10]))
    # train model
    model = train(1, train_dataset, val_dataset, test_dataset)
    # delets dataset to save memory
    del bbc

    # repeat on onion dataset for further fine-tuning
    onion = Text_DataSet('onion/data/*.txt')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(onion, util.split_proportionally(len(onion), [70, 20, 10]))
    model = train(1, train_dataset, val_dataset, test_dataset, _model=model)
    del onion
    # save the model
    model.save_pretrained('models/')
