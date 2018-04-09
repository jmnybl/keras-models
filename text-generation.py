# original version: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Model
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

def count_vocabulary_characters(filename, max_lines=100000):
    text=""
    if filename.endswith(".gz"):
        import gzip
        f=gzip.open(filename, "rt", encoding="utf-8")
    else:
        f=open(filename, "rt", encoding="utf-8")
    for i, line in enumerate(f):
        if i==max_lines:
            break
        text+=line
    chars = sorted(list(set(text))) # unique characters
    chars.insert(0,"<UNKNOWN>")
    print("First 20 items in the vocabulary:",chars[:20])
    vocabulary = dict((c, i) for i, c in enumerate(chars))
    return vocabulary

def preprocessing_characters(text, vocabulary):
    chars = sorted(list(set(text))) # unique characters
    vocabulary = dict((c, i) for i, c in enumerate(chars))
    return vocabulary, list(text)

def preprocessing_words(text):
    # TODO
    pass

def save_vocabulary(vocab, file_name):

    import json
    with open(file_name, "wt", encoding="utf-8") as f:
        print(json.dumps(vocab, indent=2), file=f)


fname="stt-data.txt.gz"

vocabulary = count_vocabulary_characters(fname) # estimate vocabulary using 100K fist lines
inversed_vocabulary = {value: key for key, value in vocabulary.items()}
print("Vocabulary size:", len(vocabulary))

save_vocabulary(vocabulary, "generation-vocab.json")


def infinite_datareader(fname, chunck_size, max_count=0):
    text_batch=[]
    counter=0
    iteration=0
    while True:
        iteration+=1
        print("Iteration:", iteration)
        if fname.endswith(".gz"):
            import gzip
            f=gzip.open(fname, "rt", encoding="utf-8")
        else:
            f=open(fname, "rt", encoding="utf-8")
        for line in f:
            text_batch+=list(line) # reads characters
            if len(text_batch)>chunck_size:
                yield text_batch, iteration
                text_batch=[]
        counter+=1
        if max_count!=0 and counter>=max_count:
            break

def infinite_vectorizer(vocabulary, fname, batch_size, context_size):
    """ Returns one batch at a time, data should be one big list. """
    X = np.zeros((batch_size, context_size)) # placeholder matrix, sequential 
    Y = np.zeros((batch_size, len(vocabulary))) # output classes, onehot
    examples=0
    for chunk, iteration in infinite_datareader(fname, 0): # this loop will never end
        step=np.random.randint(4,11) # vary the step to get slightly different examples in each iteration
        for i in range(0, len(chunk)-context_size, step):
            # this is now one example
            example=chunk[i:i+context_size]
            label=chunk[i+context_size]
            Xi=[vocabulary.get(c, vocabulary["<UNKNOWN>"]) for c in example]
            X[examples]=Xi
            Y[examples,vocabulary.get(label, vocabulary["<UNKNOWN>"])]=1
            examples+=1
            if examples==batch_size:
                yield X, Y, iteration
                X = np.zeros((batch_size, context_size)) # placeholder matrix, sequential 
                Y = np.zeros((batch_size, len(vocabulary))) # output classes, onehot
                examples=0
        
context_size=50        
embedding_size=50    
batch_size=150

# build the model
print("Building model...")

input_=Input(shape=(context_size,))
embeddings=Embedding(len(vocabulary), embedding_size)(input_)
lstm1=LSTM(512, return_sequences=True, input_shape=(context_size, embedding_size))(embeddings)
drop1=Dropout(0.2)(lstm1)
lstm2=LSTM(512, return_sequences=False)(drop1)
drop2=Dropout(0.2)(lstm2)
classification=Dense(len(vocabulary), activation="softmax")(drop2)

model=Model(inputs=[input_], outputs=[classification])

adam_optimizer = Adam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer)

print(model.summary())

import json
model_json = model.to_json()
with open("generation-model.json", "w") as f:
    print(model_json,file=f)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# save callback
#save_cb=ModelCheckpoint(filepath="generation-weights.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')


# test data
validation_generator=infinite_vectorizer(vocabulary, fname, 1, context_size)


# train the model, output generated text after each iteration
bcounter=0
best_loss=None
for X, Y, iteration in infinite_vectorizer(vocabulary, fname, batch_size, context_size):
    bcounter+=1
    loss=model.train_on_batch(X, Y)#, batch_size=batch_size, epochs=1, callbacks=[save_cb])

    if bcounter%1000==0: # report loss on every 100th batch
        print("loss:",loss, ", seen",bcounter*batch_size,"examples,",iteration,"iterations", flush=True)

    if bcounter%10000!=0: # show generation results on every 10000th batch
        continue

    # save weights
    if best_loss is None or loss<best_loss:
        print("Saving model weigths to generation-weights.h5 with loss", loss, flush=True)
        best_loss=loss
        model.save_weights("generation-weights.h5")

    # generate

    print()
    print('-' * 50)
    print('batch counter:', bcounter)

    # get next seed from validation data
    validation_X, validation_Y, dev_iteration=next(validation_generator)
    print("test:",validation_X.shape)
    print("test:",validation_X)

    generate_X=validation_X


    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        # add seed to generated
        generated = []
        for c in validation_X[0]:
            generated.append(inversed_vocabulary[int(c)])
        print('----- Generating with seed: "' + "".join(generated) + '"')
        sys.stdout.write("".join(generated))
        sentence=generated
        for i in range(200):
            # predict next character
            preds = model.predict(generate_X, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = inversed_vocabulary[next_index]
            generated += [next_char]

            # vectorize new seed
            sentence=generated[len(generated)-context_size:]
            generate_X=np.zeros((1,context_size))
            for i,c in enumerate(sentence):
                generate_X[0,i]=vocabulary.get(c,vocabulary["<UNKNOWN>"])

            # print
            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
            
print()
