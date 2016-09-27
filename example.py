import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from itertools import cycle
import time
import matplotlib.pyplot as plt
import subprocess
from utils import tsne
import pdb
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
import scipy.io as sio


def cluster(data,true_labels,n_clusters=3):

	km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
	km.fit(data)

	km_means_labels = km.labels_
	km_means_cluster_centers = km.cluster_centers_
	km_means_labels_unique = np.unique(km_means_labels)

	colors_ = cycle(colors.cnames.keys())

	initial_dim = np.shape(data)[1]
	data_2 = tsne(data,2,initial_dim,30)

	plt.figure(figsize=(12, 6))
	plt.scatter(data_2[:,0],data_2[:,1], c=true_labels)
	plt.title('True Labels')

	return km_means_labels

data_mat = sio.loadmat('wine_network.mat')
labels = sio.loadmat('wine_label.mat')
data_mat = data_mat['adjMat']
labels = labels['wine_label']
data_edge = nx.Graph(data_mat) 

with open('wine.edgelist','wb') as f:
	nx.write_weighted_edgelist(data_edge, f)

subprocess.call('~/DNGR-Keras/DNGR.py --graph_type '+'undirected'+' --input '+'wine.edgelist'+' --output '+'representation',shell=True)

df = pd.read_pickle('representation.pkl')
reprsn = df['embedding'].values
node_idx = df['node_id'].values
reprsn = [np.asarray(row,dtype='float32') for row in reprsn]
reprsn = np.array(reprsn, dtype='float32')
true_labels = [labels[int(node)][0] for node in node_idx]
true_labels = np.asarray(true_labels, dtype='int32')
cluster(reprsn,true_labels, n_clusters=3)
	
plt.show()




