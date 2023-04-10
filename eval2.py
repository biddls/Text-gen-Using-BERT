import numpy as np
from tqdm import tqdm

import bert_gen as bg
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random


path = './results/checkpoint-3000'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
model.eval()


def predict(
        sentence: [str]
):
    # tokenize corpus
    tokenized_text = tokenizer(
        sentence,
        add_special_tokens=True,
        padding='max_length',
        return_tensors='pt',
        truncation=True,
    )
    temp = tokenized_text.input_ids.detach().clone()
    # create labels
    tokenized_text['labels'] = tokenized_text.input_ids.detach().clone()

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(tokenized_text.input_ids.shape)

    # create mask array
    # doesn't mask over any PAD, CLS or SEP tokens (0, 101, 102)
    mask_arr = (rand < 0.15) * (tokenized_text.input_ids != 101) * \
               (tokenized_text.input_ids != 102) * (tokenized_text.input_ids != 0)

    selection = []

    for i in range(tokenized_text.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(tokenized_text.input_ids.shape[0]):
        tokenized_text.input_ids[i, selection[i]] = 103


    ff = (tokenized_text.input_ids == 103).nonzero()[:, 1]
    temp_segments_ids = [0] * len(tokenized_text.input_ids)
    segments_tensors = torch.tensor([temp_segments_ids])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokenized_text.input_ids, token_type_ids=segments_tensors)
        predictions = outputs[0]

    predicted_index = predictions[0, ff].argsort()[:, -1]
    _pred = predicted_index.numpy()

    # labels
    temp = (temp.numpy())[0, ff]

    # accuracy of prediction
    acc = (_pred == temp).sum()/len(_pred)

    # distance of prediction
    predicted_index = predictions[0, ff].argsort().numpy()
    predicted_index = predicted_index[:, ::-1]
    dist = np.zeros_like(temp)
    for i, j in enumerate(predicted_index):
        dist[i] = np.argwhere(temp[i] == j)

    return acc, dist

with open("./onion/NewsWebScrape.txt", 'r', encoding="UTF-8") as txt_file:
    sentences = txt_file.readlines()
sentences = [i.replace('\n', '') for i in sentences]
sentences = [i.split(' #~# ')[1] for i in sentences]

ogs = [bg.prompt_preprocessing(sentence)[0] for sentence in sentences]

metrics = [predict(og) for og in tqdm(ogs[:10])]

acc = np.mean([i[0] for i in metrics])
dist = np.mean([np.mean(i[1]) for i in metrics])

print(f"Average accuracy is: {acc * 100:.2f}%")
print(f"Average distance is: {dist:.0f} indices")
