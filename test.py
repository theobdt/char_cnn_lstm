# from torchnlp.datasets import penn_treebank_dataset
from utils.preprocessing import (
    load_params,
    clean_str,
    CharsVocabulary,
    WordsVocabulary,
)
import argparse
import os
import torch
from datetime import datetime
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "-t", "--txt_file", type=str, default="example.txt", help="Examples"
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="last",
    help="Date of the checkpoints to use or 'last'",
)
parser.add_argument(
    "--max_words",
    type=int,
    default=1,
    help="If entered, each sentence will be completed until an <eos> tag",
)
parser.add_argument(
    "--path_ckpts",
    type=str,
    default="ckpts",
    help="Path to the checkpoints folder",
)
parser.add_argument(
    "--no_unk",
    action='store_true',
    help="If entered, will not predict unknown token",
)

args = parser.parse_args()


if args.ckpt == "last":
    ckpts = os.listdir(args.path_ckpts)
    dates = [datetime.strptime(name, "%Y-%m-%d_%H-%M-%S") for name in ckpts]
    ckpt = ckpts[np.argmax(dates)]
else:
    ckpt = args.ckpt

path_model = os.path.join(args.path_ckpts, ckpt, "model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model, map_location=device)
print(f"Checkpoint {ckpt} loaded successfully")
model.eval()

path_idx2word = os.path.join(args.path_ckpts, ckpt, "data/idx2word.pkl")
path_idx2char = os.path.join(args.path_ckpts, ckpt, "data/idx2char.pkl")

char_vocab = CharsVocabulary()
char_vocab.load(path_idx2char)

word_vocab = WordsVocabulary()
word_vocab.load(path_idx2word)

path_config = os.path.join(args.path_ckpts, ckpt, "config_experiment.yaml")
path_data_params = os.path.join(args.path_ckpts, ckpt, "data/data.yaml")
config = load_params(path_config)
data_params = load_params(path_data_params)
word_length = int(data_params["word_length"])

print(f"Predicting on file {args.txt_file}")
with open(args.txt_file, "r") as txt_file:
    lines = txt_file.readlines()

cleaned_lines = [clean_str(line[:-1]).split() for line in lines]

with torch.no_grad():
    for line in cleaned_lines:
        line_str = ' '.join(line)
        hidden = model.init_hidden(1)
        encoded_words = torch.LongTensor(1, len(line), word_length)
        for i, word in enumerate(line):
            encoded_words[0, i, :] = char_vocab.to_idx(word, word_length)
        completed = []
        for _ in range(args.max_words):
            outputs, hidden = model(encoded_words, hidden)
            last_layer = outputs[0, -1, :]
            scores_idx = np.argsort(last_layer)
            # idx_next_word = np.argmax(last_layer).item()
            idx_next_word = scores_idx[-1]
            if args.no_unk and idx_next_word == 1:
                idx_next_word = scores_idx[-2]
            if idx_next_word == 0:
                break
            next_word = word_vocab.to_word(idx_next_word)
            completed.append(next_word)
            encoded_words = char_vocab.encode_word(
                next_word, word_length
            ).view(1, 1, -1)
        completed = ' ' + ' '.join(completed)
        print(f"\nInput : {line_str}")
        print(f"Output : {line_str + completed}")
