# It is implementation of NN (from scratch) of MNIST Dataset
# Download MNIST Datset and paste this file into that dataset dir

import numpy as np
import os
import mnist_loader

# Neural Netwrok architecture
layer = [784,30,10]
# Sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

# Derivative of sigmoid function
def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))

#Network Start here
def train(layer,lt=100,l_rate=0.5):
	tr_d, val_data, test_data = mnist_loader.load_data_wrapper()
	num_layers = len(layer)

	# weights and bias random initialization
	b_2 = np.random.randn(layer[1], 1)
	b_3 = np.random.randn(layer[2], 1)
	w_2 = np.random.randn(layer[1],layer[0]) #weight which connected l1 to l2
	w_3 = np.random.randn(layer[2],layer[1]) #weight which connected l2 to l3

	for e in range(lt):
		rn = random.randint(len(tr_d)+1)
		im = tr_d[rn]
		
		#input and output layer
		x = im[0]
		y = im[1]
		
		# Forward Propogation
		z_2 = np.dot(w_2,x) + b_2
		a_2 = sigmoid(z_2)	#activation of l2

		z_3 = np.dot(w_3,a_2) + b_3
		a_3 = sigmoid(z_3)	#activation of l3
		
		
		# Backpropogation start here
		delta_3 = (a_3 - y) * sigmoid_p(z_3)
		w_3 = w_3 - l_rate * delta_3 * a_2.T
		b_3 = b_3 - l_rate * delta_3

		delta_2 = (a_3 - y) * sigmoid_p(z_2)
		w_3 = w_3 - delta_3 * a_2.T
		b_3 = b_3 - delta_3
		
		dcost_pred = 2 * (pred - target)
		dpred_dz = sigmoid_p(z)
	
		dz_dw1 = point[0]
		dz_dw2 = point[1]
		dz_db = 1
	
		dcost_dw1 = dcost_pred*dpred_dz*dz_dw1
		dcost_dw2 = dcost_pred*dpred_dz*dz_dw2 
		dcost_db = dcost_pred*dpred_dz*dz_db
	
		w1 -= l_rate*dcost_dw1
		w2 -= l_rate*dcost_dw2
		b -= l_rate*dcost_db

		c = 0
		for j in range(len(data)):
			p = data[j]
			p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
			c += np.square(p_pred - p[2])

		if c < o:
			o = c
			opt = [c, w1, w2, b]
	return opt
train(layer)
