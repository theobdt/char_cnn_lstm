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

Then predict missing words with:
```
$ python3 predict.py --txt_file example.txt --n_best 3
Checkpoint ckpts/2020-05-15_20-51-11 loaded successfully
Predicting on file example.txt

Input : I saw her and she __
Prediction : I saw her and she [was/'s/is]

Input : I saw her and we __
Prediction : I saw her and we ['re/have/do]

Input : I see her and she __
Prediction : I see her and she ['s/says/is]

Input : I see her and we __
Prediction : I see her and we ['re/'ve/have]
```

## Training

We recommend training this model on GPU.
We trained it on Google Colaboratory, an example notebook can be found [here](https://colab.research.google.com/drive/1spqn7rE9du-wbxoTn7gF9tbOXsResilz?usp=sharing).

```
$ python3 train.py 
```
## Tensorboard
You can inspect checkpoints locally with tensorboard:
```
$ pip3 install tensorboard
$ tensorboard --logdir ckpts
```
