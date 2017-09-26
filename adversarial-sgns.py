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
from keras.engine.topology import Layer
from keras.callbacks import Callback
from math import ceil
from keras.preprocessing.sequence import make_sampling_table
import random

################ GRADIENT REVERSAL LAYERS #############################################
# Ganin and Lempitsky, 2015, Unsupervised Domain Adaptation by Backpropagation, ICML-15
# Implementation: @yusuke0519 commented on Jul 4, 2016 
# https://github.com/fchollet/keras/issues/3119
import theano
class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    
    @yusuke0519 commented on Jul 4, 2016 
    https://github.com/fchollet/keras/issues/3119
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes
        
class GradientReversalLayer(Layer):
    """ Reverse a gradient 
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                         "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#####################################################################

### VOCABULARY ###
class Vocabulary(object):

    def __init__(self):
    
        pass
        
        
    def invert(self,d):
        inverted_d={}
        for key,idx in d.items():
            inverted_d[idx]=key   
        return inverted_d
        
    def read_text(self,training_file,estimate):
        word_counter=0
        c_focus=Counter()
        c_target=Counter()
        tmp_focus=[]
        tmp_target=[]
        adversarials=set()
        for line in gzip.open(training_file,"rt",encoding="utf-8"):
            try:
                focus,target,label=line.strip().split("\t")
            except:
                print("Wrong number of columns:",line,file=sys.stderr)
                continue
            word_counter+=1
            tmp_focus.append(focus)
            tmp_target.append(target)
            adversarials.add(label)
            if word_counter%1000000==0:
                print(word_counter,"words",file=sys.stderr)
            if len(tmp_focus)>100000:
                c_focus.update(tmp_focus)
                c_target.update(tmp_target)
                tmp_focus=[]
                tmp_target=[]
            if estimate!=0 and word_counter>=estimate:
                break # break this file
        if len(tmp_focus)>0:
            c_focus.update(tmp_focus)
            c_target.update(tmp_target)
            
        return  c_focus, c_target, adversarials, word_counter
        
        
        
    def build(self,training_file,min_count=5,estimate=0):
        # training_file: tab separated list (focus, target, adversarial_label) of skipgram pairs, gzipped
        # min_count: discard words that appear less than X times
        # estimate: estimate vocabulary using X words, 0 for read all
        c_focus, c_target, adversarials, word_counter=self.read_text(training_file, estimate)
        focus_words={"<MASK>":0,"<UNK>":1}
        target_words={"<MASK>":0,"<UNK>":1}
        filtered_word_count=0
        for w,count in c_focus.most_common():
            if count<min_count:
                break
            focus_words[w]=len(focus_words)
            filtered_word_count+=count
        for w,count in c_target.most_common():
            if count<min_count:
                break
            target_words[w]=len(target_words)
        adv_labels={}
        for label in adversarials:
            adv_labels[label]=len(adv_labels)
        self.focus_words=focus_words
        self.target_words=target_words
        self.vocab_size=len(self.focus_words)
        self.inverted_words=self.invert(self.focus_words)
        self.adversarial_labels=adv_labels
        print("Vocabulary created with {w} words.".format(w=self.vocab_size),file=sys.stderr)
        if estimate==0 or estimate>word_counter:
            self.total_word_count=filtered_word_count
        else:
            self.total_word_count=None # unknown
            
        self.sampling_table=make_sampling_table(len(self.focus_words))
            
        
    def idx(self,word,d):
        return d.get(word,d["<UNK>"])    
        
        
        
    
    def sample_negatives(self,current_target,negatives):
        negative_samples = np.random.randint(2,len(self.target_words),negatives) # skip masking and unk
        while current_target in negative_samples:
            negative_samples = np.random.randint(2,len(self.target_words),negatives)
        return negative_samples

            
        
def infinite_iterator(fname, vs, negatives, batch_size, max_iterations=10):
    focus_words=[]
    target_words=[]
    targets=[]
    adversarial_targets=[]
    iterations=0
    while True:
        print("Iteration:",iterations,file=sys.stderr)
        #if iterations>=max_iterations: # TODO: this does not seem to be possible with fit_generator
        #    if len(focus_words)>0:
        #        yield {"focus_word":np.array(focus_words),"target_words":np.array(target_words)}, [np.array(targets),np.array(adversarial_targets)]
        #    break
        for line in gzip.open(fname,"rt",encoding="utf-8"):
            try:
                focus,target,label=line.strip().split("\t")
            except:
                print("Wrong number of columns:",line,file=sys.stderr)
                continue
            focus_word=vs.idx(focus,vs.focus_words)
            if focus_word==vs.idx("<UNK>",vs.focus_words) or vs.sampling_table[focus_word] < random.random():
                continue
#                if np.random.random_sample() < 0.8: # 80% change to drop <UNK> training example
#                    continue
            focus_words.append(focus_word)
            target_word=vs.idx(target,vs.target_words)
            negative_sample=vs.sample_negatives(target_word,negatives)

            target_words.append([target_word]+list(negative_sample))
            
            targets.append([1.0]+[0.0]*negatives)
            adversarial_targets.append(vs.adversarial_labels[label])
            if len(focus_words)==batch_size:
                    yield {"focus_word":np.array(focus_words),"target_words":np.array(target_words)}, [np.array(targets),np.array(adversarial_targets)]
                    focus_words=[]
                    target_words=[]
                    targets=[]
                    adversarial_targets=[]
        iterations+=1
 
 
class CustomCallback(Callback):

    def __init__(self,fname):
        
        self.fname=fname

    def on_epoch_end(self, epoch, logs={}):
        with open(self.fname,"a",encoding="utf-8") as f:
            print(epoch,logs.get("flatten_2_loss"),logs.get("flatten_3_loss"),sep="\t",file=f)    
      
def train(args):      
        
    # SETTINGS    
    minibatch=200
    embedding_size=args.embedding_size
    negative_size=args.negatives
    training_file=args.data
    iterations=args.iterations
    epochs=args.epochs
    

    ## VOCABULARY
    vs=Vocabulary()
    vs.build(training_file,min_count=args.min_count,estimate=args.estimate_vocabulary)
    data_iterator=infinite_iterator(training_file,vs,negative_size,minibatch)
    print(vs.adversarial_labels)
    
    # try to estimate how many steps per epoch we need to fullfill the iterations criteria (if possible)
    if iterations>0 and vs.total_word_count!=None:
        epochs=10
        steps_per_epoch=ceil((vs.total_word_count/(minibatch*2))*0.7) # estimate subsampling of frequent words drop 30% of the data
    else:
        steps_per_epoch=5000
    print("Training {e} epochs with {s} steps per epoch, and minibatch size of {m} examples.".format(e=epochs,s=steps_per_epoch,m=minibatch))
    
    ## MODEL
    
    # input
    focus_input=Input(shape=(1,), name="focus_word")
    target_input=Input(shape=(negative_size+1,), name="target_words")

    # Embeddings
    focus_embeddings=Embedding(vs.vocab_size, embedding_size, name="word_embeddings")(focus_input)
    repeated=RepeatVector(negative_size+1)(Flatten()(focus_embeddings))

    context_embeddings=Embedding(len(vs.target_words),embedding_size, name="context_embeddings")(target_input)


    def my_dot(l):
        return K.sum(l[0]*l[1],axis=-1,keepdims=True)
    
    def my_dot_dim(input_shape):
        return (input_shape[0][0],input_shape[0][1],1)

    dot_out=Lambda(my_dot,my_dot_dim)([repeated,context_embeddings])

    sigmoid_layer=Dense(1, activation='sigmoid')

    s_out=Flatten()(sigmoid_layer(dot_out))
    
    # adversarial loss
    gradient_reverse=GradientReversalLayer(1)(focus_embeddings)
    #adv_linear=Dense(200, activation='linear')(gradient_reverse)
    adversarial_prediction=Flatten()(Dense(1, activation='sigmoid')(gradient_reverse))

    model=Model(inputs=[focus_input,target_input], outputs=[s_out,adversarial_prediction])
    
    #adam=optimizers.Adam(beta_2=0.9)
    model.compile(optimizer='sgd',loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[1.0,1.0])

    print(model.summary())

    from keras.utils import plot_model

    plot_model(model,to_file=args.model_name+".png",show_shapes=True)

    # callback to print losses to a file
    custom_cb=CustomCallback(args.model_name+".stats")


    model.fit_generator(data_iterator,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[custom_cb],verbose=1)

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
    g.add_argument('-d', '--data', type=str, required=True, help='tab separated file of skipgram pairs (focus, target, adversarial_label), gzipped')
    g.add_argument('-m', '--model_name', type=str, required=True, help='Name of the saved model (in .txt format)')
    g.add_argument('--min_count', type=int, default=2, help='Frequency threshold, how many times an ngram must occur to be included? (default %(default)d)')
    g.add_argument('--embedding_size', type=int, default=200, help='Dimensionality of the trained vectors')
    g.add_argument('--negatives', type=int, default=5, help='How many negatives to sample, default=5')
    g.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    g.add_argument('--iterations', type=int, default=0, help='Iterations over training data, this overwrites the number of epochs. If iterations are used, --estimate_vocabulary must be zero. Default=0 (uses numbers of epochs).')
    g.add_argument('--estimate_vocabulary', type=int, default=0, help='Estimate vocabulary using x words, default=0 (all).')
    
    args = parser.parse_args()
    
    train(args)
