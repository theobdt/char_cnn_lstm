# from torchnlp.datasets import penn_treebank_dataset
from utils.preprocessing import (
    load_params,
    clean_str,
    CharsVocabulary,
    WordsVocabulary,
)
from utils.dataloader import DataLoader
from models.model import CharCNNLSTM
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
import pickle


parser = argparse.ArgumentParser()

parser.add_argument(
    "-t", "--txt_file", type=str, default="example.txt", help="Examples"
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="best",
    help="Date of the checkpoints to use or ['best', 'last']",
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

args = parser.parse_args()


if args.ckpt in ["best", "last"]:
    all_ckpts = os.listdir(args.path_ckpts)
    best_loss = None
    latest_date = None
    for folder in all_ckpts:
        path_params = os.path.join(args.path_ckpts, folder, "parameters.json")
        params = load_params(path_params)
        loss = params.training["best_loss"]
        date = datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
        if not latest_date or date > latest_date[0]:
            latest_date = (date, folder)
        if not best_loss or loss < best_loss[0]:
            best_loss = (loss, folder)
    if args.ckpt == "best":
        ckpt = best_loss[1]
    elif args.ckpt == "last":
        ckpt = latest_date[1]
else:
    ckpt = args.ckpt

path_model = os.path.join(args.path_ckpts, ckpt, "model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model, map_location=device)
print(f"Checkpoint {ckpt} loaded successfully")
model.eval()

path_idx2word = os.path.join(args.path_ckpts, ckpt, "idx2word.pkl")
path_idx2char = os.path.join(args.path_ckpts, ckpt, "idx2char.pkl")

char_vocab = CharsVocabulary()
char_vocab.load(path_idx2char)

word_vocab = WordsVocabulary()
word_vocab.load(path_idx2word)

path_params = os.path.join(args.path_ckpts, ckpt, "parameters.json")
model_params = load_params(path_params)
word_length = int(model_params.data["word_length"])

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
            encoded_words[0, i, :] = char_vocab.encode_word(word, word_length)
        completed = []
        for _ in range(args.max_words):
            outputs, hidden = model(encoded_words, hidden)
            last_layer = outputs[0, -1, :]
            idx_next_word = np.argmax(last_layer).item()
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
