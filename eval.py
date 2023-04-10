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
    _model (BertForMaskedLM, optional): The preloaded model to use for training. If not
        provided, the script will start training from scratch.

The script loads the input text data from a file located at 'onion/NewsWebScrape.txt',
tokenizes the data using the BERT tokenizer, and trains the model for 5 epochs.
The trained model is then saved to a file located at 'models/'.
"""

from glob import glob
from types import NoneType

import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import wandb
import torch
import util
from util import Text_DataSet
import evaluate


metric = evaluate.load("glue", "mrpc", keep_in_memory=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using: cuda' if torch.cuda.is_available() else 'Using: CPU')


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = torch.argmax(logits, dim=-1)
    # overall = None
    # for x in zip(predictions, labels):
    #
    return metric.compute(predictions=predictions, references=labels)


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
    #     do_eval=True,  # evaluate the model after training
    #     per_device_train_batch_size=1,  # batch size per device during training
    #     per_device_eval_batch_size=1,  # batch size for evaluation
    #     gradient_checkpointing=True,  # saves memory
    )

    # loads in the most recent model if it exists
    # handles checkpointing
    path = list(glob('./results/checkpoint-*'))
    path.sort(key=lambda x: int(x.split('./results\\checkpoint-')[1]))
    path = path[-1]

    _model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
    training_args.resume_from_checkpoint = path

    trainer = Trainer(
        model=_model,  # the instantiated HuggingFace Transformers model to be trained
        # args=training_args,  # training arguments, defined above
        # train_dataset=train_data,  # training dataset
        eval_dataset=test_data,  # evaluation dataset
        compute_metrics=compute_metrics,
    )
    _model.to(device)

    # evaluates the model
    _eval_results = trainer.evaluate()
    print(f"{_eval_results['eval_loss']=:.2f}")
    print(_eval_results)
    print(repr(_eval_results))
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

