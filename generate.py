# goes with text-generation.py



from __future__ import print_function
from keras.models import Model, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Embedding
from keras.layers import CuDNNLSTM as LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys

import re

### Only needed for me, not to block the whole GPU, you don't need this stuff
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
### ---end of weird stuff

## helper functions
from nltk.tokenize import word_tokenize # turns text into list of words


def load_vocabulary(file_name):

    import json
    with open(file_name, "rt", encoding="utf-8") as f:
        vocab=json.load(f)
    return vocab

def load_model(model_file, weight_file):

    with open(model_file, "rt", encoding="utf-8") as f:
        model=model_from_json(f.read())
    model.load_weights(weight_file)

    return model


vocab_file="generation-vocab.json"
model_file="generation-model.json"
weight_file="generation-weights.h5"

vocab, _ = load_vocabulary(vocab_file)
model = load_model(model_file, weight_file)

print("Vocabulary size:", len(vocab))
inversed_vocab = {value: key for key, value in vocab.items()}
print("Inversed vocabulary size:", len(inversed_vocab))
print(vocab, inversed_vocab)

        
context_size=50        
embedding_size=50    
batch_size=150


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# generate
while True:

    print()
    print('-' * 50)


    text = input("Seed for generation:").strip()

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        for c in text:
            generated.append(c)
        print('----- Generating with seed: "' + "".join(generated) + '"')
        sys.stdout.write("".join(generated))
        sentence=generated

        # vectorize seed
        generate_X=np.zeros((1,context_size))
        for i,c in enumerate(sentence):
            generate_X[0,i]=vocab.get(c,vocab["<UNKNOWN>"])

        for i in range(200):

            # predict
            preds = model.predict(generate_X, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = inversed_vocab[next_index]

            generated += [next_char]
            sentence=generated[len(generated)-context_size:]

            # vectorize new seed
            generate_X=np.zeros((1,context_size))
            for i,c in enumerate(sentence):
                generate_X[0,i]=vocab.get(c,vocab["<UNKNOWN>"])

            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
            
print()
