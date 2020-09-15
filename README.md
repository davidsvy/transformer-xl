# Transformer XL

## Table of contents

1. [ Description. ](#desc)
2. [ Datasets. ](#data)
3. [ Dependencies. ](#dep)
4. [ Usage for music. ](#music)
5. [ Usage for text. ](#text)

<a name="desc"></a>

## Description

A TensorFlow implementation of the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860). The goal of this project is to generate long and coherent sequences of data.

<a name="data"></a>

## Datasets

The neural network was trained on 2 separate datasets:

1. [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), which contains over 200 hours of virtuosic piano performances in the format of MIDI files.
2. [Fraudulent E-mail Corpus](https://www.kaggle.com/rtatman/fraudulent-email-corpus), which contains over 2,500 scam emails.

Generated data for both datasets can be found **here**

<a name="dep"></a>

## Dependencies

- NumPy
- Tensorflow
- argparse
- pathlib
- tqdm
- pickle
- re
- joblib
- mido
- glob
- bs4
- dload

<a name="music"></a>

## Usage for music

### Data Preprocessing

```
$ python preprocess_music.py -d
```

### Training

```
$ python train_music.py
```

### Music generation

```
$ python generate_music.py <n_songs> <checkpoint path>
```

<a name="text"></a>

## Usage for text

### Data Preprocessing

```
python preprocess_text.py <corpus path>
```

### Training

```
$ python train_text.py
```

### Text generation

```
$ python generate_text.py <n_samples> <checkpoint path>
```
