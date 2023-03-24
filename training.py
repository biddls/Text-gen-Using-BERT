from glob import glob
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import wandb
import torch
import util
from util import Text_DataSet
import os.path


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using: cuda' if torch.cuda.is_available() else 'Using: CPU')


def train(epochs, train_data, test_data, _model=None):
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        evaluation_strategy="steps",
        do_train=True,
        do_eval=True,  # evaluate the model after training
        num_train_epochs=epochs,  # total number of training epochs
        per_device_train_batch_size=18,  # batch size per device during training
        per_device_eval_batch_size=20,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=50,
        eval_steps=500,
        gradient_checkpointing=True,
        report_to=['wandb'],  # report results to wandb
        run_name='bert-base-uncased',  # name of run
    )

    # loads in the model if it exists
    # handles checkpointing
    path = list(glob('./results/checkpoint-*'))
    path.sort(key=lambda x: int(x.split("./results\\checkpoint-")[1]))
    path = path[-1]

    if os.path.isdir(path):
        print(f"Resuming from checkpoint. Path: {path}")
        _model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
        training_args.resume_from_checkpoint = path
    elif _model is None:
        print("Starting from scratch")
        _model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    else:
        print("Starting from pre-loaded model")

    trainer = Trainer(
        model=_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=test_data  # evaluation dataset
    )

    _model.to(device)
    trainer.train()
    # eval_results = trainer.evaluate()
    # print(f"{eval_results['eval_loss']=:.2f}")
    wandb.finish()
    return _model


if __name__ == "__main__":
    util.dataset_checks()

    # repeat on onion dataset for fine-tuning
    onion = Text_DataSet('onion/NewsWebScrape.txt', tokenizer)
    train_dataset, val_dataset = torch.utils.data.random_split(onion, util.split_proportionally(len(onion), [80, 20]))
    del onion
    model = train(20, train_dataset, val_dataset)
    # save the model
    model.save_pretrained('models/')
