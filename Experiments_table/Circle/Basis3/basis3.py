import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
import numpy as np
import operator as op
from functools import reduce
import numpy as np
import math


#AUXILIAR FUNCTIONS

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def normalize(xs):
    min_xs = np.min(xs)
    max_xs = np.max(xs)
    return (xs-min_xs)/(max_xs-min_xs)

# BASE

delta = math.pi/2-0.001
def f(t):
    return tf.math.sin((delta+t)/2)
def g(t):
    return tf.math.sin((delta-t)/2)
import scipy.special as sc

# MODEL

class first_layer(layers.Layer):
    def __init__(self, units=5,input_dim=1, activation = None):
        super(first_layer, self).__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.w = self.add_weight(shape=(1,self.units), initializer='ones',trainable=True) 
    def call(self,inputs):
        l = len(t)
        u = [ncr(n-1,i)*self.w[0][i]*(f(inputs)**i)*(g(inputs)**(n-1-i)) for i in range(n)] 
        out= tf.reshape(u/np.sum(np.array(u)),(1,self.units))
        return layers.Activation(self.activation)(out)       
class second_layer(layers.Layer):
    def __init__(self, units=5,input_dim=5, activation = None):
        super(second_layer, self).__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotNormal()
        self.p_control = self.add_weight(shape=(self.units,2), initializer=initializer,trainable=True)
        self.p0 = [train[0]]
        self.p_last = [train[-1]]
    def call(self,inputs):
        pcontrol = tf.concat([self.p0, self.p_control, self.p_last],0)
        out = tf.matmul(inputs,pcontrol)   
        return layers.Activation(self.activation)(out)    

# DATASET
l=100
t=[-delta+2*delta*(i-1)/(l-1) for i in range(1,l+1)]
bx = normalize([math.cos((i+delta)*math.pi/delta) for i in t])
by = normalize([math.sin((i+delta)*math.pi/delta) for i in t])
train=[[bx[i],by[i]] for i in range(len(bx))]

# COMPILATION


k = list()
for n in [4,5,6]:
    for i in range(5):
        inp = Input(shape=(1,))
        n_units=n
        out1 = first_layer(units=n_units)(inp)
        out2 = second_layer(units=n_units-2)(out1)
        model = Model(inputs=inp,outputs=out2)
        opt = tf.keras.optimizers.Adamax(
            learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax',
            )
        model.compile(loss='mae', optimizer=opt)

# TRAINING

        history=model.fit(
            t,train,
            batch_size=1,
            epochs=4000,
            verbose=0
            )
        k.append(history.history['loss'][-1])

# SAVING THE MODEL
    np.savetxt("history_"+str(n)+".txt",[np.min(k)])

