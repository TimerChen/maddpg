""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def normalize(inp, activation, reuse, scope, norm = 'None'):
    if norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


def construct_fc_weights(dim_input, dim_hidden, num_action):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01), name='w1')
    weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]), name='b1')
    for i in range(1,len(dim_hidden)):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([dim_hidden[i-1], dim_hidden[i]], stddev=0.01), name='w'+str(i+1))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([dim_hidden[i]]), name='b'+str(i+1))
    weights['w'+str(len(dim_hidden)+1)] = tf.Variable(tf.truncated_normal([dim_hidden[-1], num_action], stddev=0.01), name='w'+str(len(dim_hidden)+1))
    weights['b'+str(len(dim_hidden)+1)] = tf.Variable(tf.zeros([num_action]), name='b'+str(len(dim_hidden)+1))
    return weights

def forward_fc(inp, weights, num_hidden, reuse=False):
    hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
    for i in range(1,num_hidden):
        hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(num_hidden+1)]) + weights['b'+str(num_hidden+1)]

def batch2xy(batch, obs_size, num_action):
    if obs_size-batch['states'].shape[0] < 0:
        raise RuntimeError("Need longer obs_size, the real size now is: ", batch['states'].shape[0])
    batch['states'] = np.pad(batch['states'], (0,obs_size-batch['states'].shape[0]), 'constant')
    batch['next_states'] = np.pad(batch['next_states'], (0,obs_size-batch['next_states'].shape[0]), 'constant')
    batch['rewards'] = batch['rewards'].reshape([-1,1])
    x = np.concatenate([batch['states'],batch['next_states'],batch['rewards']])
    y = np.zeros((batch['action'].shape[0], num_action))
    y[:, batch['actions']] = 1
    return x, y
