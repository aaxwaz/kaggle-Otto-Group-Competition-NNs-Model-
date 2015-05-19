'''
Build a two-layer neural network where a dropout is connected
to each layer. Can use exponential decreasing learning rate 
by adding in function 'on_epoch_finished'. L2 and L1 can also 
be added provided that source code supports such feature
'''
import theano
import numpy as np
import pandas as pd
from lasagne import layers
from lasagne.layers import DropoutLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import LeakyRectify

BATCH_SIZE = 256
MAX_EPOCHS_SPACE = 150 # exponentially stretch the learning rates
MAX_EPOCHS = 3 # actual rounds of running 
LEARNING_RATE_START = 0.018 
LEARNING_RATE_END = 0.005
MOMENTUM_START = 0.9
MOMENTUM_END = 0.9

# using exponential decreasing rate 
LEARNING_RATE_START_LOG = np.log(LEARNING_RATE_START) 
LEARNING_RATE_END_LOG = np.log(LEARNING_RATE_END)
MOMENTUM_START_LOG = np.log(MOMENTUM_START) 
MOMENTUM_END_LOG = np.log(MOMENTUM_END)

INPUT_DIM = 93
NUM_OUTPUT = 9
EVAL_SIZE = 0.0 #set to 0.2 for CV; set to 0.0 for submissions 

NUM_HIDDEN_UNITS_1 = 1000
NUM_HIDDEN_UNITS_2 = 500
INPUT_DROPOUT_P = 0.15
HIDDEN1_DROPOUT_P = 0.25
HIDDEN2_DROPOUT_P = 0.25
L2_LAMBDA = 0.0000185
L1_LAMBDA = 0.0

def float32(k):
    return np.cast['float32'](k)
    
class AdjustVariable(object):
    def __init__(self, name, start, stop):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.logspace(self.start, self.stop, num = MAX_EPOCHS_SPACE, base = np.e) # num = nn.max_epochs

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

myNN = NeuralNet(
    ''' Contruct a two-hidden layers Neural Nets. With dropout ratio as 
    0.15, 0.25, 0.25. Using default ReLU as nonlinearities for each 
    activation, and softmax for output. Using decreasing learning rate
    as well as momentum. L1 and L2 are used provided source code has such
    features implemented
    '''
    layers = [
    ('input', layers.InputLayer),
    ('dropout0', layers.DropoutLayer),
    ('hidden1', layers.DenseLayer), 
    ('dropout1', DropoutLayer),
    ('hidden2', layers.DenseLayer),
    ('dropout2', layers.DropoutLayer),
    ('output', layers.DenseLayer),
    ],
    
    # layers 
    input_shape = (None, INPUT_DIM),
    dropout0_p = INPUT_DROPOUT_P, # input dropout
    hidden1_num_units = NUM_HIDDEN_UNITS_1,
    dropout1_p = HIDDEN1_DROPOUT_P, # hidden 1 dropout
    hidden2_num_units = NUM_HIDDEN_UNITS_2,
    dropout2_p = HIDDEN2_DROPOUT_P, # hidden 2 dropout
    output_num_units = NUM_OUTPUT,
    
    # Changeing rate and momentum
    update_learning_rate = theano.shared(float32(LEARNING_RATE_START)),
    update_momentum = theano.shared(float32(MOMENTUM_START)),
    
    #### changing learning rate and momentum
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=LEARNING_RATE_START_LOG, stop=LEARNING_RATE_END_LOG),
        AdjustVariable('update_momentum', start=MOMENTUM_START_LOG, stop=MOMENTUM_END_LOG),
        ],
    
    # Other params 
    batch_iterator_train = BatchIterator(batch_size = BATCH_SIZE),
    max_epochs = MAX_EPOCHS,
    update = nesterov_momentum,
    verbose = 1,
    output_nonlinearity = softmax,
    eval_size = EVAL_SIZE,
    l2_lambda = L2_LAMBDA,
    l1_lambda = L1_LAMBDA
    )
    
### main process ###
df = pd.read_csv('train.csv') # reading data ... 
df_test = pd.read_csv('test.csv')

# processing data 
df.target = df.target.apply(lambda x: x.split("_", 1)[1]) # convert class_x -> x
df.target = df.target.astype(np.int32)
df.target = df.target - 1
df = df.iloc[:,1:] 
df = df.reindex(np.random.permutation(df.index))

df_test = df_test.iloc[:,1:] # take out id col
X_test = df_test.values.astype(np.float32)

X, y = df.values[:,:93].astype(np.float32), df.values[:,93].astype(np.int32) 
    
# training 
result = myNN.fit(X,y)
    
# predict and prepare submission file 
y_prob = myNN.predict_proba(X_test) 
y_prob_df = pd.DataFrame(y_prob)
id_df = pd.DataFrame({'id':range(1,len(df_test)+1)})
final_df = pd.concat([id_df, y_prob_df], axis=1)
final_df.columns = ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
final_df.to_csv('final_result.csv', index = False)

    
    
    
    
    
    
