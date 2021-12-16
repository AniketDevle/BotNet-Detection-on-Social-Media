import pandas as pd
import numpy as np
import json
import networkx as nx
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import AgglomerativeClustering, DBSCAN

def fetchDataset(no_response_to_retrieve=-1):
    '''
    This function opens the 'SWM-dataset.csv' using pandas and
    returns the dataframe. 
    To specify the no of ids you would like to get results for, 
    use 'no_response_to_retrieve' parameter.

    Arguments: 
        no_response_to_retrieve: number of tweet ids in SWM-dataset.csv that you'd
                                 like to get results for.
    Returns: Pandas dataframe

    Example: 
    >>> no_response_to_retrieve = 100
    >>> fetchDataset(no_response_to_retrieve)
    '''

    return pd.read_csv('dataset/SWM-dataset.csv', nrows=no_response_to_retrieve)

def buildGraph(dataset):
    '''
    This function takes the dataset as an input and builds 
    the tweet-retweet graph using the 'screen-name-from' and 'screen-name-to'.
    An valid edge exists from source 'screen-name-from' and destination 'screen-name-to'.
    NOTE: All string vertex labels are converted to int labels to ease the scikit-learn.agglomerative_clustering() method.
    
    Arguments: 
        dataset: actual dataset

    Returns: Key-value pair such that key holds the 'screen-name-from' and 
             value contains the list of 'screen-name-to'
             Key-value pair containing the screen-name (string) to int label mapping

    Example: 
    >>> buildGraph(dataset)
    '''

    tweet_graph = {}
    tweet_graph_screenname_mapping = {}
    label_counter = 1
    for datarow in dataset.itertuples():
        if datarow[3] not in tweet_graph_screenname_mapping:
            tweet_graph_screenname_mapping[datarow[3]] = label_counter
            label_counter += 1
        if datarow[4] not in tweet_graph_screenname_mapping:
            tweet_graph_screenname_mapping[datarow[4]] = label_counter
            label_counter += 1
        if tweet_graph_screenname_mapping[datarow[3]] in tweet_graph:
            tweet_graph[tweet_graph_screenname_mapping[datarow[3]]].append(tweet_graph_screenname_mapping[datarow[4]])
        else:
            tweet_graph[tweet_graph_screenname_mapping[datarow[3]]] = [tweet_graph_screenname_mapping[datarow[4]]]
    return tweet_graph, tweet_graph_screenname_mapping

def computeKClique(tweet_graph):
    '''
    This function takes the tweet_gragh as an input and computes 
    number of communities in the graph using the k-clique algorithm.
    For our usage, k=2
    
    Arguments: 
        tweet_graph: the tweet graph built in buildGraph()

    Returns: list of nodes which form communities

    Example: 
    >>> computeKClique(tweet_graph)
    '''

    networkx_tweet_graph = nx.Graph(tweet_graph)
    graph_communities = list(k_clique_communities(networkx_tweet_graph, 2))
    return graph_communities

def computeModularity(tweet_graph):
    '''
    This function takes the tweet_gragh as an input and computes 
    number of communities in the graph using the Clauset-Newman-Moore greedy modularity maximization.
    
    Arguments: 
        tweet_graph: the tweet graph built in buildGraph()

    Returns: list of nodes which form communities

    Example: 
    >>> computeModularity(tweet_graph)
    '''

    networkx_tweet_graph = nx.Graph(tweet_graph)
    graph_communities = list(greedy_modularity_communities(networkx_tweet_graph))
    return graph_communities

def computeAgglomerativeClustering(tweet_graph, n_clusters=2):
    '''
    This function takes the tweet_gragh as an input and computes 
    number of clusters in the graph using agglomerative clustering.
    
    Arguments: 
        tweet_graph: the tweet graph built in buildGraph()
        n_clusters: The number of clusters to find

    Returns: Agglomerative clustering

    Example: 
    >>> computeAgglimerativeClustering(tweet_graph, 6)
    '''

    tweet_graph_2d = []
    for v1 in tweet_graph:
        for v2 in tweet_graph[v1]:
            tweet_graph_2d.append([v1, v2])
    tweet_graph_np = np.array(tweet_graph_2d)
    graph_agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(tweet_graph_np)
    return graph_agglomerative_clustering

def computeDbscanClustering(tweet_graph, eps=10):
    '''
    This function takes the tweet_gragh as an input and computes 
    number of clusters in the graph using the DBSCAN algorithm.
    
    Arguments: 
        tweet_graph: the tweet graph built in buildGraph()
        eps: Max distance between two nodes

    Returns: DBSCAN clustering

    Example: 
    >>> computeDbscanClustering(tweet_graph, eps=6)
    '''

    tweet_graph_2d = []
    for v1 in tweet_graph:
        for v2 in tweet_graph[v1]:
            tweet_graph_2d.append([v1, v2])
    tweet_graph_np = np.array(tweet_graph_2d)
    graph_dbscan_clustering = DBSCAN(eps=10).fit(tweet_graph_np)
    return graph_dbscan_clustering

if __name__ == '__main__':
    dataset = fetchDataset(1000)
    tweet_graph, tweet_graph_screenname_mapping = buildGraph(dataset)
    graph_communities_kclique = computeKClique(tweet_graph)
    graph_communities_modularity = computeModularity(tweet_graph)
    # graph_agglomerative_clustering = computeAgglimerativeClustering(tweet_graph, 6)
    graph_dbscan_clustering = computeDbscanClustering(tweet_graph, eps=10)
