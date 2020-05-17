import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import numpy as np
import sys

from utils.objects import load_params, get_data_params
from utils.vocab import CharsVocabulary, WordsVocabulary
from utils.dataloader import DataLoader
from utils.preprocessing import preprocess_sentence


parser = argparse.ArgumentParser()

parser.add_argument("-t", "--txt_file", type=str, help="Examples")
parser.add_argument(
    "--n_best",
    type=int,
    default=1,
    help="Number of best predicted words shown",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="last",
    help="Date of the checkpoints to use or 'last'",
)
parser.add_argument(
    "--path_ckpts",
    type=str,
    default="ckpts",
    help="Path to the checkpoints folder",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Evaluate on test set",
)

args = parser.parse_args()

if not (args.txt_file or args.test):
    print("ERROR: no action requested, please select 'txt_file' or 'test'")
    sys.exit(-1)

if args.ckpt == "last":
    ckpts = [
        folder
        for folder in os.listdir(args.path_ckpts)
        if os.path.isdir(os.path.join(args.path_ckpts, folder))
    ]
    dates = [datetime.strptime(name, "%Y-%m-%d_%H-%M-%S") for name in ckpts]
    ckpt = ckpts[np.argmax(dates)]
    path_ckpt = os.path.join(args.path_ckpts, ckpt)
else:
    path_ckpt = args.ckpt

path_model = os.path.join(path_ckpt, "model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model, map_location=device)
print(f"Checkpoint {path_ckpt} loaded successfully")
model.eval()

path_idx2word = os.path.join(path_ckpt, "data/idx2word.pkl")
path_idx2char = os.path.join(path_ckpt, "data/idx2char.pkl")

char_vocab = CharsVocabulary()
char_vocab.load(path_idx2char)

word_vocab = WordsVocabulary()
word_vocab.load(path_idx2word)

path_config = os.path.join(path_ckpt, "config_experiment.yaml")
path_data_params = os.path.join(path_ckpt, "data/data.yaml")
config = load_params(path_config)

if args.test:
    print('Evaluating on test set')
    print(f"Corpus name : {config['data']['corpus_name']}")
    data_params, path_objects = get_data_params(config["data"])
    word_vocab_size = data_params["word_vocab_size"]
    char_vocab_size = data_params["char_vocab_size"]

    seq_len = config["training"]["seq_len"]
    test_loader = DataLoader(path_objects, "test", 1, seq_len)
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in tqdm(range(len(test_loader))):
            inputs, targets = test_loader[i]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # repackaging hidden state
            hidden = tuple([h.detach().to(device) for h in hidden])

            outputs, hidden = model(inputs, hidden)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, word_vocab_size), targets
            )
            total_loss += len(outputs) * loss.item()
        avg_loss = total_loss / len(test_loader)
        avg_perplexity = np.exp(avg_loss)
        print(f"Test set: CE={avg_loss:.3f}, PPL={avg_perplexity:.2f}")

if args.txt_file:
    print(f"Predicting on file {args.txt_file}")
    with open(args.txt_file, "r") as txt_file:
        raw_lines = [line.split() for line in txt_file.readlines()]

    preprocessed_lines = [preprocess_sentence(line) for line in raw_lines]
    data_params = load_params(path_data_params)
    word_length = data_params["computed_params"]["word_length"]
    with torch.no_grad():
        for i, line in enumerate(preprocessed_lines):
            hidden = model.init_hidden(1)
            predictions = raw_lines[i].copy()
            output = None
            for j, word in enumerate(line):
                if word == "__":
                    if output is not None:
                        best_words = word_vocab.output2word(
                            output, n_best=args.n_best
                        )
                        predictions[j - 1] = f"[{'/'.join(best_words)}]"
                        word = best_words[0]
                    else:
                        raise ValueError("Output tensor not initialized")
                encoded_word = char_vocab.to_idx(word, word_length)
                encoded_word = encoded_word.view(1, 1, -1)
                output, hidden = model(encoded_word, hidden)
            print(f"\nInput : {' '.join(raw_lines[i])}")
            print(f"Prediction : {' '.join(predictions)}")
