import tensorflow as tf
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import function
from collections import namedtuple
import numpy as np
import sys
import os
from datetime import datetime
import input_data_eager as input_data
import tensorflow_probability as tfp
from six.moves import xrange
import tensorflow.keras.backend as K

units = 40

state0 = tf.zeros(( 2, 50, 200))

def keras_step(input_b, prev_state):
    z = prev_state
    i_in = tf.matmul(input_b, M)
    new_z = z + i_in
    return new_z

a = tf.zeros((1,5))

M = tf.Variable(initial_value=tf.random.truncated_normal(shape=(40,200), mean=0.0, stddev= 0.5))

I = tf.Variable(initial_value=tf.random.truncated_normal(shape=(50, 80, 40), mean=0.0, stddev= 0.5))

#Z = tf.matmul(I,M)

#z = keras_step(I,state0)
#print(z)
print(tuple(state0))
o = K.rnn(step_function=keras_step, inputs=I, initial_states=state0)
print(o)
b = tf.zeros((1,5))
#print(M,I)
#print(Z)

