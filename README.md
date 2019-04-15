# romanesco

A vanilla recurrent neural network (RNN) language model. Supports model
training, text scoring, and text generation.

## Installation

Make sure you have an NVIDIA GPU at your disposal, with all drivers and CUDA
installed. Make sure you also have `python >= 3.5`, `pip` and `git` installed,
and run

```bash
git clone https://github.com/zurichnlp/romanesco.git
cd romanesco
pip install --user -e .
```

If you have sudo privileges and prefer to install `romanesco` for all users on
your system, omit the `--user` flag. The `-e` flag installs the app in “editable
mode”, meaning you can change source files (such as `romanesco/const.py`) at any
time.

## Model training

Models are trained from a single plaintext file with one sentence per line.
Symbols – e.g., words or characters – are delimited by blanks.

Example input (word-level):

```
I love the people of Iowa .
So that &apos;s the way it is .
Very simple .
```

Example input (character-level):

```
I <blank> l o v e <blank> t h e <blank> p e o p l e <blank> o f <blank> I o w a .
S o <blank> t h a t &apos; s <blank> t h e <blank> w a y <blank> i t <blank> i s .
V e r y <blank> s i m p l e .
```

`romanesco` doesn't preprocess training data. If you want to train a model on lowercased input, for example, you'll need to lowercase the training data yourself.

To train a model from `corpus.train.txt` using GPU 0, run

```bash
CUDA_VISIBLE_DEVICES=0 romanesco train corpus.train.txt
```

By default, the trained model and vocabulary will be stored in a directory called `model`, and logs (for monitoring with Tensorboard) in `logs`. You can use custom destinations through the `-m` and `-l` command line arguments, respectively. Folders will be created if they don't exist.

Some hyperparameters can be adjusted from the command line; run `romanesco train -h` for details. Other hyperparameters are currently hardcoded in `romanesco/const.py`.


## Scoring

Once you've trained a model, you can use it to score texts. `romanesco` will calculate the [perplexity](https://en.wikipedia.org/wiki/Perplexity) of a text given a trained model. Lower is better: if you've trained a model on TV subtitles, it will typically assign lower scores to other TV subtitles than, say, an article from the New York Times.

To score `my-article.txt` using GPU 0, run

```bash
CUDA_VISIBLE_DEVICES=0 romanesco score my-article.txt
```

This assumes there is a folder called `model` in your current working directory, containing a model trained with `romanesco` (see above). If your model is stored somewhere else, use the `-m` command line argument.

For further options, run `romanesco score -h`.

## Sampling

A trained model can be used to generate new text resembling the original training data. To generate a text with length 200 (number of symbols), run

```bash
CUDA_VISIBLE_DEVICES=0 romanesco sample 200
```

This assumes there is a folder called `model` in your current working directory, containing a model trained with `romanesco` (see above). If your model is stored somewhere else, use the `-m` command line argument.

For further options, run `romanesco sample -h`.
