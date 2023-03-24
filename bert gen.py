import torch
from transformers import BertTokenizer, BertForMaskedLM
import sacremoses
import random

# path = glob('./results/checkpoint-*')[-1]
path = './results/checkpoint-3000'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
model.eval()
deTok = sacremoses.MosesDetokenizer('en')


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def predict(
        sentence: ([str], [str]),
        _sentence_length: int,
        liveOutput: bool = False
):
    baseSentence, fullSentence = sentence[0], sentence[1]
    if baseSentence == fullSentence:
        print("\n###\n")
        print(fullSentence)
        print("\n###\n")

    filler = ' '.join(['MASK' for _ in range(int(_sentence_length))])

    if len(fullSentence.strip()) == 0:
        sentence = "[CLS] " + filler + " . [SEP]"
    else:
        sentence = "[CLS] " + fullSentence + " " + filler + " . [SEP]"

    tokenized_text = tokenizer.tokenize(sentence)
    idxs = duplicates(tokenized_text, 'mask')
    for masked_index in idxs:
        tokenized_text[masked_index] = "[MASK]"

    # LOOP TO CREATE TEXT
    generated = 0
    first = None
    while generated < int(_sentence_length):
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
        # print(predicted_token)
        tokenized_text[focus_mask_idx] = predicted_token
        generated += 1

        if liveOutput:
            sentence = tokenizer.convert_tokens_to_string(tokenized_text)
            sentence = sentence.replace("[CLS] ", "").replace(" [SEP]", "")
            sentence = deTok.detokenize(sentence.split(" "))
            sentence = sentence.split('[MASK]', maxsplit=1)[0]

            if first is None:
                first = " ".join(sentence.split(" ")[:-2])

            print(colored(0, 255, 0, first), end='')
            print(colored(255, 0, 0, sentence[len(first)+1:]), end='\r')

    sentence = tokenizer.convert_tokens_to_string(tokenized_text)
    sentence = sentence.replace("[CLS] ", "").replace(" [SEP]", "")
    sentence = deTok.detokenize(sentence.split(" "))
    return sentence


def prompt_preprocessing(_sentence: str, max_length: int = 200, floorDiv: int = 4) -> (str, str):
    sentence = _sentence.split(" ")
    _length = min(len(sentence) // floorDiv, max_length)
    sentence = sentence[:_length]
    sentence = " ".join(sentence)
    return _sentence, sentence, _length


if __name__ == "__main__":
    with open("./onion/NewsWebScrape.txt", 'r', encoding="UTF-8") as txt_file:
        sentences = txt_file.readlines()
    sentences = [i.replace('\n', '') for i in sentences]
    sentences = [i.split(' #~# ')[1] for i in sentences]
    sentence_length = 50
    og, sentence_orig, length = prompt_preprocessing(random.choice(sentences))
    sentence_length = min(length, sentence_length)

    out = predict((og, sentence_orig), sentence_length, liveOutput=True)
