import torch
import random
from transformers import BertTokenizer, BertModel
from glob import glob

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

print("\n###\n")

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def predict():
    sentence = ""
    a = random.choice(list(glob('bbc/**/*.txt')))
    with open(a, "r") as txt_file:
        sentence_orig = txt_file.readlines()
    sentence_orig = " ".join(sentence_orig).replace('\n', '')
    sentence_orig = sentence_orig.split(" ")
    sentence_orig = sentence_orig[:len(sentence_orig) // 4]
    sentence_orig = " ".join(sentence_orig)

    print(sentence_orig)
    print("\n###\n")

    sentence_length = 100
    filler = ' '.join(['MASK' for _ in range(int(sentence_length))])

    if len(sentence_orig.strip()) == 0:
        sentence = "[CLS] " + filler + " . [SEP]"
    else:
        sentence = "[CLS] " + sentence_orig + " " + filler + " . [SEP]"

    tokenized_text = tokenizer.tokenize(sentence)
    idxs = duplicates(tokenized_text, 'mask')
    for masked_index in idxs:
        tokenized_text[masked_index] = "[MASK]"

    ##### LOOP TO CREATE TEXT #####
    generated = 0
    full_sentence = []
    while generated < int(sentence_length):
        mask_idxs = duplicates(tokenized_text, "[MASK]")

        focus_mask_idx = min(mask_idxs)

        mask_idxs.pop(mask_idxs.index(focus_mask_idx))
        temp_tokenized_text = tokenized_text.copy()
        temp_tokenized_text = [j for i, j in enumerate(temp_tokenized_text) if i not in mask_idxs]
        temp_indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
        ff = [idx for idx, i in enumerate(temp_indexed_tokens) if i == 103]
        temp_segments_ids = [0] * len(temp_tokenized_text)
        tokens_tensor = torch.tensor([temp_indexed_tokens])
        segments_tensors = torch.tensor([temp_segments_ids])

        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        # TOP - k Sampling
        k = 5
        predicted_index = random.choice(predictions[0, ff].argsort()[0][-k:]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token)
        tokenized_text[focus_mask_idx] = predicted_token
        generated += 1

    return ' '.join(tokenized_text[1:-1]).replace('[ review ]', '')

if __name__ == "__main__":
    print(predict())