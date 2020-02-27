# Character-aware neural language model

Pytorch implementation of the paper [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)

## Installation
This project requires python >= 3.5

```
$ git clone https://github.com/theobdt/char_cnn_lstm.git
$ pip3 install -r requirements.txt
```


## Inference

First, download pre-trained models:
```
$ chmod +x download_model.sh
$ ./download_model.sh
```

Then predict next 10 words with:
```
$ python3 test.py --max_words 10
```


## Training

We recommend training this model on GPU.
We trained it on Google Colaboratory.

```
$ python3 train.py --n_epochs 30
```

If you are using Google Colab and want to save weights on google drive :
```
$ python3 train.py --n_epochs 30 --gdrive /content/gdrive
```
