# from torchnlp.datasets import penn_treebank_dataset
from utils.preprocessing import initialize_dataset, load_params
from utils.dataloader import DataLoader
from models.model import CharCNNLSTM
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "-bs", "--batch_size", type=int, default=20, help="Batch size"
)
parser.add_argument(
    "-ds",
    "--dataset",
    type=str,
    default="penn-treebank",
    help="Name of the dataset to use",
)
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument(
    "-l",
    "--seq_len",
    type=int,
    default=35,
    help="Length of the words sequence used for training",
)
parser.add_argument(
    "-reset",
    "--reset_objects",
    action="store_true",
    help="If entered, words and characters objects are recomputed",
)
parser.add_argument(
    "-mw",
    "--max_words",
    type=int,
    default=None,
    help="Maximum size of the words vocabulary",
)
parser.add_argument(
    "-mwl",
    "--max_words_len",
    type=int,
    default=None,
    help="Maximum size of words",
)
parser.add_argument(
    "--lr", type=float, default=1, help="Start learning rate",
)
parser.add_argument(
    "--n_epochs", type=int, default=25, help="Number of epochs",
)
parser.add_argument(
    "--log_interval", type=int, default=200, help="Number of epochs",
)
parser.add_argument(
    "--save", type=str, default="ckpts", help="Number of epochs",
)
parser.add_argument(
    "--model", type=str, default=None, help="Name of the model",
)
parser.add_argument(
    "--clip_grad", type=float, default=5, help="Clip value of gradient",
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="LSTM dropout",
)
args = parser.parse_args()

path_objects = os.path.join("data", args.dataset, "objects")
if not os.path.exists(path_objects) or args.reset_objects:
    initialize_dataset(args.dataset, args.max_words_len, args.max_words)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use : {device}")

if args.model:
    path_ckpt = os.path.join(args.save, args.model)
else:
    path_ckpt = os.path.join(
        args.save, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
os.makedirs(path_ckpt, exist_ok=True)
print(f"Initialized checkpoint {path_ckpt}")
path_model = os.path.join(path_ckpt, "model.pt")

train_loader = DataLoader(args.dataset, "train", args.batch_size, args.seq_len)
val_loader = DataLoader(args.dataset, "val", 1, args.seq_len)
data_params = load_params(args.dataset)

size_char_vocab = data_params.size_char_vocab
size_word_vocab = data_params.size_word_vocab
lr = args.lr

model = CharCNNLSTM(
    char_vocab_size=size_char_vocab,
    char_embedding_size=15,
    word_vocab_size=size_word_vocab,
    filters_width=[1, 2, 3, 4, 5, 6],
    num_filters=[25] * 6,
    num_layers=2,
    hidden_size=300,
    dropout=args.dropout
).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


def train(loader):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    for i in range(len(loader)):
        optimizer.zero_grad()
        inputs, targets = loader[i]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # repackaging hidden state
        hidden = tuple([h.detach().to(device) for h in hidden])

        outputs, hidden = model(inputs, hidden, debug=False)
        loss = loss_function(outputs.view(-1, size_word_vocab), targets)
        # print(f'training loss : {loss}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            print(f"Training step {i + 1}/{len(loader)}, loss: {avg_loss}")
            total_loss = 0


def evaluate(loader):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(len(loader)):
            inputs, targets = loader[i]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # repackaging hidden state
            hidden = tuple([h.detach().to(device) for h in hidden])

            outputs, hidden = model(inputs, hidden, debug=False)
            loss = loss_function(outputs.view(-1, size_word_vocab), targets)
            # print(f'training loss : {loss}')
            print(
                f"step {i + 1}/{len(loader)}, validation loss: {loss}",
                end="\r",
            )
            total_loss += len(outputs) * loss.item()
    print("")
    return total_loss / len(loader)


best_val_loss = None
for i in range(args.n_epochs):
    print(f"\nEpoch {i+1}/{args.n_epochs}")
    train_loss = train(train_loader)
    val_loss = evaluate(val_loader)
    print(f"Validation: CE={val_loss}, PPL={np.exp(val_loss)}")
    if not best_val_loss or val_loss < best_val_loss:
        with open(path_model, "wb") as f:
            torch.save(model, f)
        print(f"Model saved to {path_model}")
        best_val_loss = val_loss
    else:
        lr /= 2
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print(f'Learning rate reduced to {lr}')
