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


def colored(
        r: int,
        g: int,
        b: int,
        text: str
) -> str:
    """
    Adds color to the given text using ANSI escape codes.

    Parameters:
    r (int): Red color value.
    g (int): Green color value.
    b (int): Blue color value.
    text (str): Text to color.

    Returns:
    str: The colored text.
    """
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def duplicates(
        lst: list,
        item: str
) -> list[int]:
    """
    Returns the indices of all occurrences of a given item in a list.

    Parameters:
    lst (list): The list to search.
    item (str): The item to search for.

    Returns:
    list[int]: A list of indices where the item occurs in the list.
    """
    return [i for i, x in enumerate(lst) if x == item]


def outputPredictions(tokenized_text: [str], first: str):
    """
    Prints the generated text with custom coloured formatting to see what's been generated and what's original.

    Parameters:
        tokenized_text (list of str): A list of tokenized strings.
        first (str): A string that will be used to determine what text is original and what text has been generated.

    Returns:
        None
    """

    sentence = tokenizer.convert_tokens_to_string(tokenized_text)
    sentence = sentence.replace("[CLS] ", "").replace(" [SEP]", "")
    sentence = deTok.detokenize(sentence.split(" "))
    sentence = sentence.split('[MASK]', maxsplit=1)[0]

    if first is None:
        first = " ".join(sentence.split(" ")[:-2])

    # Print original text in green and generated text in red
    print(colored(0, 255, 0, first), end='')
    print(colored(255, 0, 0, sentence[len(first) + 1:]), end='\r')

    return first


def predict(
        sentence: ([str], [str]),
        _sentence_length: int,
        liveOutput: bool = False
) -> str:
    """
    Predicts the next words in a sentence using the BERT model with masked language modeling.

    Parameters:
    sentence (tuple([str], [str])): A tuple containing the original sentence and a shortened version for prediction.
    _sentence_length (int): The number of words to predict.
    liveOutput (bool): Whether to display live output.

    Returns:
    str: The completed sentence.
    """
    baseSentence, fullSentence = sentence[0], sentence[1]
    # If the full and base sentences are the same, print the full sentence
    if baseSentence == fullSentence:
        print("\n###\n")
        print(fullSentence)
        print("\n###\n")

    # Create a string of MASK tokens of length _sentence_length
    filler = ' '.join(['MASK' for _ in range(int(_sentence_length))])

    # Create the input sentence to the BERT model
    if len(fullSentence.strip()) == 0:
        sentence = "[CLS] " + filler + " . [SEP]"
    else:
        sentence = "[CLS] " + fullSentence + " " + filler + " . [SEP]"

    # Tokenize the input sentence
    tokenized_text = tokenizer.tokenize(sentence)
    # Replace all 'mask' tokens with '[MASK]' tokens
    for masked_index in duplicates(tokenized_text, 'mask'):
        tokenized_text[masked_index] = "[MASK]"

    # Loop to create the predicted text
    generated = 0
    first = None
    # Loop until the predicted text is the same length as the original text
    while generated < int(_sentence_length):
        maskIds = duplicates(tokenized_text, "[MASK]")
        focus_mask_idx = min(maskIds)

        maskIds.pop(maskIds.index(focus_mask_idx))
        temp_tokenized_text = tokenized_text.copy()
        temp_tokenized_text = [j for i, j in enumerate(temp_tokenized_text) if i not in maskIds]
        temp_indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
        ff = [idx for idx, i in enumerate(temp_indexed_tokens) if i == 103]
        temp_segments_ids = [0] * len(temp_tokenized_text)
        tokens_tensor = torch.tensor([temp_indexed_tokens])
        segments_tensors = torch.tensor([temp_segments_ids])

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        # TOP - k Sampling
        k = 5
        predicted_index = random.choice(predictions[0, ff].argsort()[0][-k:]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        tokenized_text[focus_mask_idx] = predicted_token
        generated += 1

        if liveOutput:
            first = outputPredictions(tokenized_text, first)

    # Convert the tokenized text back to a string
    sentence = tokenizer.convert_tokens_to_string(tokenized_text)
    sentence = sentence.replace("[CLS] ", "").replace(" [SEP]", "")
    sentence = deTok.detokenize(sentence.split(" "))
    return sentence, tokenized_text


def prompt_preprocessing(
        _sentence: str,
        max_length: int = 200,
        floorDiv: int = 4
) -> (str, str, int):
    """
    Preprocesses the input sentence to prepare it for prompt-based text generation.

    Args:
        _sentence (str): The input sentence to be preprocessed.
        max_length (int): The maximum length of the preprocessed sentence (default: 200).
        floorDiv (int): The division factor used to reduce the length of the input sentence (default: 4).

    Returns:
        tuple: A tuple containing the following elements:
            - _sentence (str): The original input sentence.
            - sentence (str): The preprocessed sentence.
            - _length (int): The length of the preprocessed sentence.
    """
    # Split the input sentence into individual words
    sentence = _sentence.split(" ")

    # Determine the length of the preprocessed sentence based on the floor division of the length of the input sentence
    _length = min(len(sentence) // floorDiv, max_length)

    # Take the first _length words and join them together with spaces to form the preprocessed sentence
    sentence = " ".join(sentence[:_length])

    # Return a tuple containing the original input sentence, the preprocessed sentence,
    # and the length of the preprocessed sentence
    return _sentence, sentence, _length


if __name__ == "__main__":
    with open("./onion/NewsWebScrape.txt", 'r', encoding="UTF-8") as txt_file:
        sentences = txt_file.readlines()
    sentences = [i.replace('\n', '') for i in sentences]
    sentences = [i.split(' #~# ')[1] for i in sentences]
    sentence_length = 50
    og, sentence_orig, length = prompt_preprocessing(random.choice(sentences))
    sentence_length = min(length, sentence_length)

    out, tokens = predict((og, sentence_orig), sentence_length, liveOutput=True)
