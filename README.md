# Transformer XL

## Table of contents

1.  [Description](#desc)
2.  [Model Architecture](#arch)

3.  [Dataset](#data)

4.  [Dependencies](#dep)

5.  [Usage for music](#music)

6.  [Usage for text](#text)

<a  name="desc"></a>

## Description

The goal of this project is to generate long and coherent sequences of data using Transformer architectures based on the following papers:

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)
- [Music Transformer](https://arxiv.org/abs/1809.04281)
- [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444)

The neural networks are tested on two separate tasks : music generation and text generation. All the models are implemented from scratch in Tensorflow 2.

<a  name="arch"></a>

## Model Architecture

![image](https://github.com/davidsvy/transformer-xl/raw/master/readme/music_model.png)

![image](https://github.com/davidsvy/transformer-xl/raw/master/readme/text_model.png)

<a  name="data"></a>

## Dataset

For the task of music generation the union of the following datasets is used:

1.  [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
2.  [SMD MIDI-Audio Piano Music](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi)
3.  [Stanford University Piano Roll Archive](https://github.com/pianoroll/SUPRA)
4.  [Classical Music ML Format](https://www.kaggle.com/jembishop1/classical-music-piano-rolls)

All of the above contain classical piano music in MIDI format. The MIDI files are preprocessed with the [mido](https://mido.readthedocs.io/en/latest/) library.

---

As for the text generation, the [CLAIR collection of "Nigerian" fraud emails](https://www.kaggle.com/rtatman/fraudulent-email-corpus) is used.

---

Generated data for both datasets can be found [here](https://github.com/davidsvy/transformer-xl/tree/master/generated_samples).

<a  name="dep"></a>

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

<a  name="music"></a>

## Usage for music

### Data Preprocessing

```

python preprocess_music.py -d

```

### Training

```

python train_music.py

```

### Music generation

```

python generate_music.py <n_songs> <checkpoint path>

```

<a  name="text"></a>

## Usage for text

### Data Preprocessing

```

python preprocess_text.py <corpus path>

```

### Training

```

python train_text.py

```

### Text generation

```

python generate_text.py <n_samples> <checkpoint path>

```
