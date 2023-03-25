"""
Trains a masked language model using the BERT architecture on a given corpus of text data,
and saves the trained model for future use.

Usage:
    python train.py

The script uses the transformers library for loading the pre-trained BERT model,
tokenizing the input text, and training the model.

The train() function trains the model on the input data and returns the trained model.
The function takes the following arguments:

    epochs (int): The number of epochs to train the model for.
    train_data (Text_DataSet): The training dataset.
    test_data (Text_DataSet): The validation dataset.
    _model (BertForMaskedLM, optional): The pre-loaded model to use for training. If not
        provided, the script will start training from scratch.

The script loads the input text data from a file located at 'onion/NewsWebScrape.txt',
tokenizes the data using the BERT tokenizer, and trains the model for 5 epochs.
The trained model is then saved to a file located at 'models/'.
"""

from glob import glob
from types import NoneType
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import wandb
import torch
import util
from util import Text_DataSet
import os.path


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using: cuda' if torch.cuda.is_available() else 'Using: CPU')


def train(
        epochs: int,
        train_data: torch.utils.data.Dataset,
        test_data: torch.utils.data.Dataset,
        _model: BertForMaskedLM | NoneType = None
) -> (BertForMaskedLM, dict[str, float]):
    """
    Trains a BERT model on a given dataset.

    Args:
        epochs (int): Number of epochs to train the model for.
        train_data (torch.utils.data.Dataset): Training dataset.
        test_data (torch.utils.data.Dataset): Evaluation dataset.
        _model (BertForMaskedLM | NoneType): Model to train. If None, uses HuggingFace's BERT.

    Returns:
        tuple: A tuple containing the trained model and the evaluation results.
            - The trained BERT model (BertForMaskedLM).
            - A dictionary of evaluation results, including the evaluation loss (dict).
    """
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
        logging_steps=50, # number of steps between logging
        eval_steps=500, # number of steps between evaluations
        gradient_checkpointing=True, # saves memory
        report_to=['wandb'],  # report results to wandb
        run_name='bert-base-uncased',  # name of run
    )

    # loads in the most recent model if it exists
    # handles checkpointing
    path = list(glob('./results/checkpoint-*'))
    path.sort(key=lambda x: int(x.split('./results\\checkpoint-')[1]))
    path = path[-1]

    # if a model is passed in, use that
    if os.path.isdir(path):
        print(f'Resuming from checkpoint. Path: {path}')
        _model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
        training_args.resume_from_checkpoint = path
    # if a model is not passed in, use the one from HuggingFace
    elif _model is None:
        print('Starting from scratch')
        _model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # if a model cannot be found raise an error
    else:
        raise FileNotFoundError(f'The model at {path} does not exist.')

    trainer = Trainer(
        model=_model,  # the instantiated HuggingFace Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=test_data  # evaluation dataset
    )

    # transfers the model to the GPU if available
    _model.to(device)
    # trains the model
    trainer.train()
    # evaluates the model
    _eval_results = trainer.evaluate()
    print(f"{_eval_results['eval_loss']=:.2f}")

    wandb.finish()
    return _model, _eval_results


if __name__ == "__main__":
    corpusPath = 'onion/NewsWebScrape.txt'
    util.dataset_checks(corpusPath)

    # repeat on onion dataset for fine-tuning
    onion = Text_DataSet(corpusPath, tokenizer)
    # noinspection PyUnresolvedReferences
    train_dataset, val_dataset = torch.utils.data.random_split(onion, util.split_proportionally(len(onion), [80, 20]))
    del onion
    model, eval_results = train(5, train_dataset, val_dataset)
    # save the model
    model.save_pretrained('models/')
