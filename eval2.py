import os
import numpy as np
from tqdm import tqdm
import bert_gen as bg
import torch
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

path = './results/checkpoint-3000'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
model.to(device)
model.eval()
print("Imports and loads complete")


def predict(
        tokenized_text: torch.Tensor,
) -> [float, np.ndarray]:
    temp = tokenized_text.detach().clone()
    if temp.shape == ():
        raise ValueError("tokenized_text must be a tensor of shape (1, 512)")

    # create mask array
    # doesn't mask over any PAD, CLS or SEP tokens (0, 101, 102)
    counter = 5
    ff = torch.tensor([0])
    while ff.shape[0] <= 1:
        mask_arr = (torch.rand(tokenized_text.shape) < 0.15) * \
                   (tokenized_text != 101) * \
                   (tokenized_text != 102) * \
                   (tokenized_text != 0)

        selection = mask_arr.nonzero()[:, 1]

        tokenized_text[0, selection] = 103

        # gets indices of the mask tokens
        ff = (tokenized_text == 103).nonzero()[:, 1]
        counter -= 1

        if counter == 0:
            return np.array([])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokenized_text.to(device))
        outputs = outputs[0].cpu()

    # predicted index
    predicted_index = outputs[0, ff].argsort()
    _predicted_index = predicted_index[:, -1]
    _pred = _predicted_index.numpy()

    # labels
    temp = (temp.numpy())[0, ff]

    # distance of prediction
    predicted_index = predicted_index.numpy()[:, ::-1]
    _dist = np.zeros_like(temp)
    for i, j in enumerate(predicted_index):
        _dist[i] = np.argwhere(temp[i] == j)

    return _dist


if __name__ == "__main__":
    # checks if dataset has been cached
    if os.path.exists("ogs.pt"):
        # load dataset from file
        ogs = torch.load("ogs.pt")
        print("Loaded dataset from file")
    else:
        # cashing to file
        with open("./onion/NewsWebScrape.txt", 'r', encoding="UTF-8") as txt_file:
            sentences = txt_file.readlines()
        sentences = [i.replace('\n', '') for i in sentences]
        sentences = [i.split(' #~# ')[1] for i in sentences]

        ogs = [bg.prompt_preprocessing(sentence)[0] for sentence in sentences]

        ogs = tokenizer(
            ogs,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        ).input_ids
        ogs = ogs.reshape(ogs.shape[0], 1, -1)
        torch.save(ogs, "ogs.pt")


    metrics = [predict(sentence) for sentence in tqdm(ogs, desc="Predicting")]

    # save metrics
    with open("metrics.csv", 'a') as f:
        for dist in tqdm(metrics, desc="Saving metrics"):
            f.write(f"{','.join(dist.astype(str))}\n")

    dist = np.mean([np.mean(i) for i in metrics])

    print(f"Average distance is: {dist:.0f} indices")
