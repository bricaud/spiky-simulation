import numpy as np
import pysad.exploration
from pysad.NodeInfo import SynthNodeInfo
from collections import Counter
import networkx as nx

def ball_test(graph_handle,params):
	node_dic = {}
	for it in range(params['nb_iter']):
		print('-- experiment',it,'--')
		if params['initial_node'] is None:
			# random number at each iteration
			initial_node = np.random.randint(graph_handle.G.number_of_nodes()-1)
		else:
			initial_node = params['initial_node']
		print('Initial node', initial_node)
		total_node_list, total_nodes_df, total_edges_df, node_acc = pysad.exploration.spiky_ball([initial_node], 
																	graph_handle, 
																	exploration_depth=params['exploration_depth'],
																	mode='percent',
																	random_subset_size=params['random_subset_size'],
																	balltype=params['balltype'],
																	node_acc=SynthNodeInfo(),
																	number_of_nodes=params['number_of_nodes'])

		# Record in which iteration a node was visited 
		for node in total_node_list:
			if node in node_dic:
				node_dic[node].append(it) 
			else:
				node_dic[node] = [it]
	subgraph = nx.from_pandas_edgelist(total_edges_df)

	# delete the initial node (always in the list)
	#del node_dic[params['initial_node']] 
	return node_dic,subgraph

def expand_degrees(node_dic,degree_dic):
	degree_list = []
	for node, occur_list in node_dic.items():
		for n in range(len(occur_list)):
			degree_list.append(degree_dic[node])
	return degree_list

# Visualization of the degree distribution

def drop_zeros(a_list):
	return [i for i in a_list if i>0]

def log_binning(counter_dict,bin_count=35):
	""" log-binning found there: 
		https://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions
	"""

	max_x = np.log10(max(counter_dict.keys()))
	max_y = np.log10(max(counter_dict.values()))
	max_base = max([max_x,max_y])

	min_x = np.log10(min(drop_zeros(counter_dict.keys())))

	bins = np.logspace(min_x,max_base,num=bin_count)

	# Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
	bin_means_y = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.values()))[0] /
				   np.histogram(list(counter_dict.keys()),bins)[0])
	bin_means_x = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.keys()))[0] /
				   np.histogram(list(counter_dict.keys()),bins)[0])

	return bin_means_x,bin_means_y

def degree_distribution(degree_list, mode='log', density=True):
	#ba_c = nx.degree_centrality(ba_g)

	count_dic = dict(Counter(degree_list))#.values()))

	if mode == 'lin':    # linear bins
		dd_x,dd_y = list(count_dic.keys()),list(count_dic.values())
	elif mode == 'log':# log bins
		dd_x,dd_y = log_binning(count_dic,20)
	else:
		raise('Unknown mode, use mode="lin" or mode="log".')
	
	if density == True: # normalize
		dd_y = [v / len(degree_list) for v in dd_y]
	return dd_x,dd_y  