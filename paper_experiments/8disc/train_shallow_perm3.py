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

SITES = 8
VARS = 4
HIDDEN = 128

invar = T.tensor3() 
targ = T.tensor3()

input = lasagne.layers.InputLayer((None, VARS, SITES), input_var = invar)

# Define subnetwork for 1st layer
dinp_1 = lasagne.layers.InputLayer((None,2*VARS,SITES,SITES))
dense1_1 = lasagne.layers.NINLayer(dinp_1, num_units = HIDDEN)

# Define subnetwork for 2nd layer
dinp2 = lasagne.layers.InputLayer((None,2*HIDDEN,SITES,SITES))
dense1_2 = lasagne.layers.NINLayer(dinp2, num_units = HIDDEN)

# Define subnetwork for 3rd layer
dinp3 = lasagne.layers.InputLayer((None,2*HIDDEN,SITES,SITES))
dense1_3 = lasagne.layers.NINLayer(dinp3, num_units = VARS, nonlinearity = None)

perm1 = PermutationalLayer(input, subnet = dense1_1)
perm2 = PermutationalLayer(perm1, subnet = dense1_2)
output = PermutationalLayer(perm2, subnet = dense1_3)

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
	err = train(seq[:-10],seq[10:]) # Predict 10 timesteps into the future = 0.2
	
	f = open("error_shallow_perm3.txt","a")
	f.write("%d %.6g\n" % (epoch,err))
	f.close()
	
	pickle.dump(lasagne.layers.get_all_param_values(output),open("network_shallow_perm3.params","wb"))
