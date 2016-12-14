import numpy as np
import matplotlib.pyplot as plt

from math import *

from PIL import Image
import glob

import pickle


import lasagne
from lasagne.layers import helper
import theano
import theano.tensor as T

from permutationlayer import PermutationalLayer
from simulate import doSimulation

SITES = 12
VARS = 4
HIDDEN = 128

invar = T.tensor3() 
targ = T.tensor3()

input = lasagne.layers.InputLayer((None, VARS, SITES), input_var = invar)

reshape = lasagne.layers.ReshapeLayer(input, (-1,VARS*SITES))
dense1 = lasagne.layers.DenseLayer(input, num_units = HIDDEN)
dense2 = lasagne.layers.DenseLayer(dense1, num_units = HIDDEN)
dense3 = lasagne.layers.DenseLayer(dense2, num_units = HIDDEN)
dense4 = lasagne.layers.DenseLayer(dense3, num_units = VARS*SITES, nonlinearity = None)

ressum = lasagne.layers.ElemwiseSumLayer([reshape, dense4])
output = lasagne.layers.ReshapeLayer(ressum,(-1,VARS,SITES))

out = lasagne.layers.get_output(output)
loss = T.mean( (out - targ)**2 )
#reg = lasagne.regularization.regularize_network_params(output, lasagne.regularization.l2)*1e-5

lr = theano.shared( np.cast['float32'](1e-3))
params = lasagne.layers.get_all_params(output,trainable=True)
updates = lasagne.updates.adam(loss,params,learning_rate = lr)

train = theano.function([invar, targ], loss, updates=updates, allow_input_downcast=True)
predict = theano.function([invar], out, allow_input_downcast = True)

# Train network

for epoch in range(20000):
	lr.set_value( np.cast['float32'](1e-3*exp(-epoch/5000.0)))
	seq = doSimulation(SITES,200,400)
	print seq.shape
	err = train(seq[:-10],seq[10:]) # Predict 10 timesteps into the future = 0.2
	
	f = open("denseskip4_error.txt","a")
	f.write("%d %.6g\n" % (epoch,err))
	f.close()
	
	pickle.dump(lasagne.layers.get_all_param_values(output),open("denseskip4_network.params","wb"))
