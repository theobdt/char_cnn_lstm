from utils.preprocessing import (
    initialize_dataset,
    load_params,
    zipfolder,
)
from utils.dataloader import DataLoader
from models.model import CharCNNLSTM
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import shutil


parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="cfg/config_experiment.yaml",
    help="Path to the config file for the experiment",
)
parser.add_argument(
    "-reset",
    "--reset_objects",
    action="store_true",
    help="If entered, words and characters objects are recomputed",
)
parser.add_argument(
    "--debug", action="store_true",
)
parser.add_argument(
    "--log_interval", type=int, default=50, help="Logging interval",
)
parser.add_argument(
    "--path_ckpts",
    type=str,
    default="ckpts",
    help="Path to checkpoints folder",
)
parser.add_argument(
    "--path_logs",
    type=str,
    default="ckpts",
    help="Path to the tensorboard directory",
)
parser.add_argument(
    "--gdrive",
    type=str,
    help=(
        "Location to mount google drive to. Checkpoints will be saved on "
        "your drive"
    ),
)
args = parser.parse_args()
config = load_params(args.config)

path_ckpts = args.path_ckpts
path_logs = args.path_logs
if args.gdrive:
    from google.colab import drive

    drive.mount(args.gdrive)
    path_ckpts = os.path.join(
        args.gdrive, "My Drive/char_cnn_lstm", args.path_ckpts
    )
    path_logs = os.path.join(
        args.gdrive, "My Drive/char_cnn_lstm", args.path_logs
    )

path_objects = os.path.join("data", config["data"]["corpus"], "objects")
if not os.path.exists(path_objects) or args.reset_objects:
    initialize_dataset(**config["data"])


path_data_params = os.path.join(path_objects, "data.yaml")
data_params = load_params(path_data_params)

# Initialize checkpoint
ckpt_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

path_ckpt = os.path.join(path_ckpts, ckpt_name)
path_ckpt_data = os.path.join(path_ckpt, "data")
path_model = os.path.join(path_ckpt, "model.pt")
os.makedirs(path_ckpt_data)
path_ckpt_config = os.path.join(path_ckpt, "config.yaml")

path_idx2char = os.path.join(path_objects, "idx2char.pkl")
path_idx2word = os.path.join(path_objects, "idx2word.pkl")

shutil.copy2(path_idx2char, path_ckpt_data)
shutil.copy2(path_idx2word, path_ckpt_data)
shutil.copy2(path_data_params, path_ckpt_data)
shutil.copy2(args.config, path_ckpt)
print(f"Initialized checkpoint {path_ckpt}")

# create data loaders
batch_size = config["training"]["batch_size"]
seq_len = config["training"]["seq_len"]
train_loader = DataLoader(path_objects, "train", batch_size, seq_len)
val_loader = DataLoader(path_objects, "val", 1, seq_len)

char_vocab_size = data_params["char_vocab_size"]
word_vocab_size = data_params["word_vocab_size"]

# instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use : {device}")
model = CharCNNLSTM(
    char_vocab_size=char_vocab_size,
    word_vocab_size=word_vocab_size,
    **config["network"],
).to(device)

# define loss and optimizer
loss_function = nn.CrossEntropyLoss()
lr = config["optimizer"]["learning_rate"]
clip_grad = config["optimizer"]["clip_grad"]
optimizer = optim.SGD(model.parameters(), lr=lr)

# initialize tensorboard and save model graph
writer = SummaryWriter(os.path.join(path_logs, ckpt_name))
sample_input, _ = train_loader[0]
sample_hidden = model.init_hidden(batch_size)
sample_hidden = tuple([h.detach().to(device) for h in sample_hidden])
writer.add_graph(model, (sample_input.to(device), sample_hidden))


def train(loader, global_step):
    model.train()
    total_loss = 0
    current_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(len(loader)):
        optimizer.zero_grad()
        inputs, targets = loader[i]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # repackaging hidden state
        hidden = tuple([h.detach().to(device) for h in hidden])

        outputs, hidden = model(inputs, hidden, debug=args.debug)
        loss = loss_function(outputs.view(-1, word_vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        current_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % args.log_interval == 0:
            avg_loss = current_loss / args.log_interval
            print(f"Training step {i + 1}/{len(loader)}, loss: {avg_loss}")
            current_loss = 0
            # tensorboard logging
            writer.add_scalar("train_loss", avg_loss, global_step + i)
            writer.add_scalar(
                "train_perplexity", np.exp(avg_loss), global_step + i
            )

    return total_loss / len(loader)


def evaluate(loader, global_step):
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

            outputs, hidden = model(inputs, hidden, debug=args.debug)
            loss = loss_function(outputs.view(-1, word_vocab_size), targets)
            total_loss += len(outputs) * loss.item()
        avg_loss = total_loss / len(loader)
        avg_perplexity = np.exp(avg_loss)
        print(f"Validation: CE={avg_loss}, PPL={avg_perplexity}")
        writer.add_scalar("val_loss", avg_loss, global_step)
        writer.add_scalar("val_perplexity", avg_perplexity, global_step)
        writer.add_scalar("learning_rate", lr, global_step)
    return avg_loss


best_perplexity = None
n_epochs = config["training"]["n_epochs"]
for i in range(n_epochs):
    print(f"\nEpoch {i+1}/{n_epochs}")
    train(train_loader, i * len(train_loader))
    val_loss = evaluate(val_loader, (i + 1) * len(train_loader))
    val_perplexity = np.exp(val_loss)
    if not best_perplexity:
        best_perplexity = val_perplexity
    if val_perplexity < best_perplexity:
        if val_perplexity > best_perplexity - 1:
            lr /= 2
            optimizer = optim.SGD(model.parameters(), lr=lr)
            print(f"Learning rate reduced to {lr}")
        best_perplexity = val_perplexity
        with open(path_model, "wb") as f:
            torch.save(model, f)
        print(f"Model saved to {path_model}")
writer.close()

if args.gdrive:
    path_zip = os.path.join(path_ckpts, f"{ckpt_name}.zip")
    zipfolder(path_ckpt, path_zip)
    print(f"Zipfile saved to {path_zip}")
