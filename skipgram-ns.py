from keras.models import Model
from keras.layers import Dense, Input, Reshape, Flatten, dot, RepeatVector, concatenate
from keras.layers.embeddings import Embedding
from keras import optimizers
import sys
import gzip
import numpy as np
import keras.backend as K
from keras.layers.core import Lambda
from collections import Counter


class Vocabulary(object):

    def __init__(self,data=None):
    
        self.words=None
        
        
    def build(self,training_file,min_count=5,estimate=0):
        # min_count: discard words that appear less than X times
        # estimate: estimate vocabulary using X words, 0 for read all
        word_counter=0
        c=Counter()
        tmp=[]
        for line in gzip.open(training_file,"rt",encoding="utf-8"):
            for word in line.strip().split(" "):
                word_counter+=1
                tmp.append(word)
                if word_counter%1000000==0:
                    print(word_counter,"words",file=sys.stderr)
            if len(tmp)>100000:
                c.update(tmp)
                tmp=[]
            if estimate!=0 and word_counter>=estimate:
                break
        if len(tmp)>0:
            c.update(tmp)
        words={"<MASK>":0,"<UNK>":1}
        for w,count in c.most_common():
            if count<min_count:
                break
            words[w]=len(words)
        self.words=words
        self.vocab_size=len(self.words)
        self.inverted_words={}
        for key,idx in self.words.items():
            self.inverted_words[idx]=key
        print("Vocabulary created with {w} words.".format(w=self.vocab_size),file=sys.stderr)
        self.total_word_count=word_counter
            
        
    def word_idx(self,word):
        return self.words.get(word,self.words["<UNK>"])    
        
        
    def make_sampling_table():
        # TODO: make a proper sampling table with probabilities and all
        pass
        
    
    def sample_negatives(self,current_word,negatives):
        negative_samples = np.random.randint(1,self.vocab_size,negatives)
        while current_word in negative_samples:
            negative_samples = np.random.randint(1,self.vocab_size,negatives)
        return negative_samples
        
        
def infinite_iterator(fname, vs, window, negatives, batch_size, max_iterations=10):
    focus_words=[]
    target_words=[]
    targets=[]
    iterations=0
    while True:
        print("Iteration:",iterations,file=sys.stderr)
        if iterations==max_iterations:
            if len(examples)>0:
                yield {"focus_word":np.array(focus_words),"target_words":np.array(target_words)}, np.array(targets)
            break
        for line in gzip.open(fname,"rt",encoding="utf-8"):
            words=line.strip().split(" ")
            for i in range(0,len(words)): # i is a focus word now
                focus_word=vs.word_idx(words[i])
                if focus_word==vs.word_idx("<UNK>"):
                    if np.random.random_sample() < 0.8: # 80% change to drop <UNK> training example
                        continue
                for j in range(max(0,i-window),min(len(words),i+window+1)):
                    if i==j:
                        continue
                    target_word=vs.word_idx(words[j])
                    negative_sample=vs.sample_negatives(focus_word,negatives)
                    focus_words.append(focus_word)
                    target_words.append([target_word]+list(negative_sample))
                    targets.append([1.0]+[0.0]*negatives)
                    if len(focus_words)==batch_size:
                        yield {"focus_word":np.array(focus_words),"target_words":np.array(target_words)}, np.array(targets)
                        focus_words=[]
                        target_words=[]
                        targets=[]
        iterations+=1
    
      
def train(args):      
        
    # SETTINGS    
    minibatch=400
    embedding_size=args.embedding_size
    window_size=args.window
    negative_size=args.negatives
    training_file=args.data
    steps_per_epoch=10000

    ## VOCABULARY
    vs=Vocabulary()
    vs.build(training_file,min_count=args.min_count,estimate=args.estimate_vocabulary)
    data_iterator=infinite_iterator(training_file,vs,window_size,negative_size,minibatch)
    
    ## MODEL
    
    # input
    focus_input=Input(shape=(1,), name="focus_word")
    target_input=Input(shape=(negative_size+1,), name="target_words")

    # Embeddings
    focus_embeddings=Embedding(vs.vocab_size, embedding_size, name="word_embeddings")(focus_input)
    repeated=RepeatVector(negative_size+1)(Flatten()(focus_embeddings))

    context_embeddings=Embedding(vs.vocab_size,embedding_size, name="context_embeddings")(target_input)


    def my_dot(l):
        return K.sum(l[0]*l[1],axis=-1,keepdims=True)
    
    def my_dot_dim(input_shape):
        return (input_shape[0][0],input_shape[0][1],1)

    dot_out=Lambda(my_dot,my_dot_dim)([repeated,context_embeddings])

    sigmoid_layer=Dense(1, activation='sigmoid')

    s_out=Flatten()(sigmoid_layer(dot_out))


    model=Model(inputs=[focus_input,target_input], outputs=[s_out])
    
    adam=optimizers.Adam(beta_2=0.9)
    model.compile(optimizer=adam,loss='binary_crossentropy')

    print(model.summary())

    from keras.utils import plot_model

    plot_model(model,to_file=args.model_name+".png",show_shapes=True)


    model.fit_generator(data_iterator,steps_per_epoch=steps_per_epoch,epochs=args.epochs,verbose=1)

    def save_embeddings(model, name, vs):
        # save word embeddings and vocabulary
            with open(name, 'wt') as f:
                embedding_w=model.get_weights()[0]
                print(vs.vocab_size,embedding_size,file=f)
                for i in range(0,vs.vocab_size):
                    print(vs.inverted_words[i]," ".join(str(x) for x in embedding_w[i]),file=f)
                
                
    save_embeddings(model,args.model_name,vs)
            
            
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-d', '--data', type=str, required=True, help='Training data file (gzipped)')
    g.add_argument('-m', '--model_name', type=str, required=True, help='Name of the saved model (in .txt format)')
    g.add_argument('--min_count', type=int, default=2, help='Frequency threshold, how many times an ngram must occur to be included? (default %(default)d)')
    g.add_argument('--window', type=int, default=5, help='Window size, default is 5 words to the left and 5 words to the right')
    g.add_argument('--embedding_size', type=int, default=200, help='Dimensionality of the trained vectors')
    g.add_argument('--negatives', type=int, default=5, help='How many negatives to sample, default=5')
    g.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    g.add_argument('--estimate_vocabulary', type=int, default=0, help='Estimate vocabulary using x words, default=0 (all).')
    
    args = parser.parse_args()
    
    train(args)            
            
