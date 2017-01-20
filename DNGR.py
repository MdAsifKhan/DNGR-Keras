#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, noise
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import networkx as nx
import pandas as pd
from utils import DataGenerator, tsne
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import sys
import pdb


def read_graph(filename,g_type):
	with open('data/'+filename,'rb') as f:
		if g_type == "undirected":
			G = nx.read_weighted_edgelist(f)
  		else:
			G = nx.read_weighted_edgelist(f,create_using=nx.DiGraph())
		node_idx = G.nodes()
	adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None,weight='weight').todense())	
	return adj_matrix, node_idx

def scale_sim_mat(mat):
	# Scale Matrix by row
	mat  = mat - np.diag(np.diag(mat))
	D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
	mat = np.dot(D_inv,  mat)

	return mat

def random_surfing(adj_matrix,max_step,alpha):
	# Random Surfing
	nm_nodes = len(adj_matrix)
	adj_matrix = scale_sim_mat(adj_matrix)
	P0 = np.eye(nm_nodes, dtype='float32')
	M = np.zeros((nm_nodes,nm_nodes),dtype='float32')
	P = np.eye(nm_nodes, dtype='float32')
	for i in range(0,max_step):
		P = alpha*np.dot(P,adj_matrix) + (1-alpha)*P0
		M = M + P

	return M

def PPMI_matrix(M):

	M = scale_sim_mat(M)
	nm_nodes = len(M)

	col_s = np.sum(M, axis=0).reshape(1,nm_nodes)
	row_s = np.sum(M, axis=1).reshape(nm_nodes,1)
	D = np.sum(col_s)
	rowcol_s = np.dot(row_s,col_s)
	PPMI = np.log(np.divide(D*M,rowcol_s))
	PPMI[np.isnan(PPMI)] = 0.0
	PPMI[np.isinf(PPMI)] = 0.0
	PPMI[np.isneginf(PPMI)] = 0.0
	PPMI[PPMI<0] = 0.0

	return PPMI


def model(data, hidden_layers, hidden_neurons, output_file, validation_split=0.9):


	train_n = int(validation_split * len(data))
	batch_size = 50
	train_data = data[:train_n,:]
	val_data = data[train_n:,:]

	input_sh = Input(shape=(data.shape[1],))
	encoded = noise.GaussianNoise(0.2)(input_sh)
	for i in range(hidden_layers):
		encoded = Dense(hidden_neurons[i], activation='relu')(encoded)
		encoded = noise.GaussianNoise(0.2)(encoded)

	decoded = Dense(hidden_neurons[-2], activation='relu')(encoded)
	for j in range(hidden_layers-3,-1,-1):
		decoded = Dense(hidden_neurons[j], activation='relu')(decoded)
	decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

	autoencoder = Model(input=input_sh, output=decoded)
	autoencoder.compile(optimizer='adadelta', loss='mse')

	checkpointer = ModelCheckpoint(filepath='data/bestmodel' + output_file + ".hdf5", verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

	train_generator = DataGenerator(batch_size)
	train_generator.fit(train_data, train_data)
	val_generator = DataGenerator(batch_size)
	val_generator.fit(val_data, val_data)

	autoencoder.fit_generator(train_generator,
		samples_per_epoch=len(train_data),
		nb_epoch=100,
		validation_data=val_generator,
		nb_val_samples=len(val_data),
		max_q_size=batch_size,
		callbacks=[checkpointer, earlystopper])
	enco = Model(input=input_sh, output=encoded)
	enco.compile(optimizer='adadelta', loss='mse')
	reprsn = enco.predict(data)
	return reprsn


def process_scripts(args):
	
	filename = args.input
	graph_type = args.graph_type
	Ksteps = args.random_surfing_steps
	alpha = args.random_surfing_rate
	output_file = args.output
	hidden_layers = args.hidden_layers
	hidden_neurons = args.neurons_hiddenlayer

	data_mat, node_idx = read_graph(filename, graph_type)
	data = random_surfing(data_mat, Ksteps, alpha)
	data = PPMI_matrix(data)

	reprsn = model(data,hidden_layers,hidden_neurons,output_file)
	data_reprsn = {'embedding':list(reprsn),'node_id':node_idx}
	df = pd.DataFrame(data_reprsn)
	df.to_pickle('data/'+output_file+'.pkl')

def main():
  parser = ArgumentParser('DNGR',
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument('--graph_type', default='undirected',
                      help='Undirected or directed graph as edgelist')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument('--random_surfing_steps', default=10, type=int,
                      help='Number of steps for random surfing')

  parser.add_argument('--random_surfing_rate', default=0.98, type=float,
                      help='alpha random surfing')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--hidden_layers', default=3, type=int,
                      help='AutoEnocoder Layers')

  parser.add_argument('--neurons_hiddenlayer', default=[128,64,32], type=list,
                      help='Number of Neurons AE.')

  args = parser.parse_args()

  process_scripts(args)

if __name__ == '__main__':
	sys.exit(main())	
	
