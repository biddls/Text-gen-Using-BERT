import numpy as np
from tqdm import tqdm
import bert_gen as bg
import torch
from transformers import BertTokenizer, BertForMaskedLM


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
    ).input_ids
    temp = tokenized_text.detach().clone()

    # create mask array
    # doesn't mask over any PAD, CLS or SEP tokens (0, 101, 102)
    mask_arr = (torch.rand(tokenized_text.shape) < 0.15) * \
               (tokenized_text != 101) * \
               (tokenized_text != 102) * \
               (tokenized_text != 0)

    selection = mask_arr.nonzero()[:, 1]

    tokenized_text[0, selection] = 103

    # gets indices of the mask tokens
    ff = (tokenized_text == 103).nonzero()[:, 1]

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokenized_text)

    # predicted index
    predicted_index = outputs[0][0, ff].argsort()
    _predicted_index = predicted_index[:, -1]
    _pred = _predicted_index.numpy()

    # labels
    temp = (temp.numpy())[0, ff]

    # accuracy of prediction
    _acc = np.sum(_pred == temp)/len(_pred)

    # distance of prediction
    predicted_index = predicted_index.numpy()[:, ::-1]
    _dist = np.zeros_like(temp)
    for i, j in enumerate(predicted_index):
        _dist[i] = np.argwhere(temp[i] == j)

    return _acc, _dist


if __name__ == "__main__":
    with open("./onion/NewsWebScrape.txt", 'r', encoding="UTF-8") as txt_file:
        sentences = txt_file.readlines()
    sentences = [i.replace('\n', '') for i in sentences]
    sentences = [i.split(' #~# ')[1] for i in sentences]

    ogs = [bg.prompt_preprocessing(sentence)[0] for sentence in sentences][:10]

    metrics = [predict(sentence) for sentence in tqdm(ogs)]

    # save metrics
    with open("metrics.csv", 'a') as f:
        for acc, dist in metrics:
            f.write(f"{acc},{','.join(dist.astype(str))}\n")

    acc = np.mean([i[0] for i in metrics])
    dist = np.mean([np.mean(i[1]) for i in metrics])

    print(f"Average accuracy is: {acc * 100:.2f}%")
    print(f"Average distance is: {dist:.0f} indices")
