from models.model import CharCNNLSTM
from utils.objects import load_params
import torch
from torch.utils.tensorboard import SummaryWriter


def test_model():
    config = load_params("cfg/config_experiment.yaml")
    char_vocab_size = 49
    word_vocab_size = 1000
    model = CharCNNLSTM(
        char_vocab_size=char_vocab_size,
        word_vocab_size=word_vocab_size,
        **config["network"]
    )
    hidden = model.init_hidden(16)
    sample_input = torch.randint(0, char_vocab_size, (16, 35, 19))
    print(sample_input.shape)

    writer = SummaryWriter("tests/logs")
    writer.add_graph(model, (sample_input, hidden))
    writer.close()
    return True
