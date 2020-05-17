import pytest
from torchnlp.datasets import penn_treebank_dataset

from utils.vocab import WordsVocabulary, CharsVocabulary


@pytest.fixture
def sample_data():
    return penn_treebank_dataset(dev=True)


def test_WordsVocabulary(sample_data):
    words_vocabulary = WordsVocabulary()
    for word in sample_data:
        words_vocabulary.add_word(word)
    words_vocabulary.sort()

    for word in sample_data:
        idx = words_vocabulary.to_idx(word)
        decoded_word = words_vocabulary.to_word(idx)
        assert word == decoded_word


def test_CharsVocabulary(sample_data):

    chars_vocabulary = CharsVocabulary()
    for word in sample_data:
        chars_vocabulary.add_word(word)
    chars_vocabulary.process()

    for word in sample_data:
        idx_tensor = chars_vocabulary.to_idx(word)
        chars = chars_vocabulary.to_chars(idx_tensor.int().tolist())
        assert chars == list(word)
