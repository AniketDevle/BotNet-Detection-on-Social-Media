import community as community_louvain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite


def build_graph(shares_df, coor_shares_df, percentile_edge_weight=90, timestamps=90):
    coord_df = coor_shares_df[['share_date', 'retweet_tid', 'screen_name_from']].reset_index(drop=True)
    coord_graph = nx.from_pandas_edgelist(coord_df, 'screen_name_from', 'retweet_tid', create_using=nx.DiGraph)

    # remove self loop nodes
    coord_graph.remove_edges_from(nx.selfloop_edges(coord_graph))

    # build bipartite graph
    screen_name_froms = list(coor_shares_df['screen_name_from'].unique())
    retweet_tids = list(coor_shares_df['retweet_tid'].unique())
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(retweet_tids, bipartite=0)
    bipartite_graph.add_nodes_from(screen_name_froms, bipartite=1)

    for index, row in coord_df.iterrows():
        bipartite_graph.add_edge(row['screen_name_from'], row['retweet_tid'], share_date=row['share_date'])

    # graph projection with account nodes
    full_graph = bipartite.weighted_projected_graph(bipartite_graph, screen_name_froms)

    shares_gb = shares_df.reset_index().groupby(['screen_name_from'])

    account_info_df = shares_gb['index'].agg([('shares', 'count')])
    account_info_df = account_info_df.merge(
        pd.DataFrame(shares_gb['is_coordinated'].apply(lambda x: (x == True).sum())).rename(
            columns={'is_coordinated': 'coord_shares'}), left_index=True, right_index=True)
    account_info_df = account_info_df.reset_index().rename(columns={'screen_name_from': 'screen_name_from'})

    # filter the dataframe with the graph nodes
    node_info_df = account_info_df[account_info_df['screen_name_from'].isin(list(full_graph.nodes))]

    attributes = []
    for node in full_graph.nodes():
        records = node_info_df[node_info_df['screen_name_from'] == node]
        attributes.append(node)
        attributes.append({
            'shares': records['shares'].values[0],
            'coord_shares': records['coord_shares'].values[0]
        })

    # update graph attributes
    it = iter(attributes)
    nx.set_node_attributes(full_graph, dict(zip(it, it)))

    # set the percentile_edge_weight number of repetedly coordinated link sharing to keep
    q = np.percentile([d['weight'] for (u, v, d) in full_graph.edges(data=True)], percentile_edge_weight)

    # create a new graph where node degree > 0
    highly_connected_graph = full_graph.subgraph([key for (key, value) in full_graph.degree if value > 0]).copy()

    # remove where the edge weitght is less than the given percentile value
    edges_to_remove = [(u, v) for (u, v, d) in highly_connected_graph.edges(data=True) if d['weight'] < q]
    highly_connected_graph.remove_edges_from(edges_to_remove)
    highly_connected_graph.remove_nodes_from(list(nx.isolates(highly_connected_graph)))

    if timestamps:
        print('Calculating nodes timestamps')
        vec_func = np.vectorize(lambda u, v: bipartite_graph.get_edge_data(u, v)['share_date'])
        attributes = []
        for (u, v) in highly_connected_graph.edges():
            attributes.append((u, v))
            attributes.append({"timestamp_coord_share": vec_func(
                np.intersect1d(list(list(bipartite_graph.neighbors(u))), list(list(bipartite_graph.neighbors(v)))), u)})

        it = iter(attributes)
        nx.set_edge_attributes(highly_connected_graph, dict(zip(it, it)))
        print("timestamps calculated")

    # find and annotate nodes-components
    connected_components = list(nx.connected_components(highly_connected_graph))
    components_df = pd.DataFrame(
        {"node": connected_components, "component": [*range(1, len(connected_components) + 1)]})
    components_df['node'] = components_df['node'].apply(lambda x: list(x))
    components_df = components_df.explode('node')

    # add cluster to simplyfy the analysis of large components
    cluster_df = pd.DataFrame(community_louvain.best_partition(highly_connected_graph).items(),
                              columns=['node', 'cluster'])

    # re-calculate the degree on the graph
    degree_df = pd.DataFrame(list(highly_connected_graph.degree()), columns=['node', 'degree'])

    # sum up the edge weights of the adjacent edges for each node
    strength_df = pd.DataFrame(list(highly_connected_graph.degree(weight='weight')), columns=['node', 'strength'])

    attributes_df = components_df.merge(cluster_df, on='node').merge(degree_df, on='node').merge(strength_df, on='node')

    # update graph attribues
    nx.set_node_attributes(highly_connected_graph, attributes_df.set_index('node').to_dict('index'))
    print('graph built')

    return highly_connected_graph, q


def get_estimated_threshold(datarows):
    ''' Estimates a threshold in seconds that defines a coordinated link share. Here threshold is calculated as a function of the median co-share time difference. More specifically, the function ranks all
        co-shares by time-difference from first share and focuses on the behaviour of the quickest second share performing q\% (default 0.5) retweets
        The value returned is the median time in seconds spent by these URLs to cumulate the p\% (default 0.1) of their total shares.'''

    # count #of different original tweets
    orig_df = pd.DataFrame(datarows['retweet_tid'].value_counts())
    orig_df.reset_index(level=0, inplace=True)
    orig_df.columns = ['retweet_tid', "share_count"]
    # filter the retweets where count>1
    orig_df = orig_df[orig_df["share_count"] > 1]
    # filter those retweet_ids from original dataframe
    datarows = datarows[datarows.set_index('retweet_tid').index.isin(orig_df.set_index('retweet_tid').index)]
    # metrics creation
    # converting object to datetime
    datarows['postedtime'] = datarows['postedtime'].astype('datetime64[ns]')
    ranks_df = datarows[['retweet_tid', 'postedtime']]
    grouped_tweets = datarows.groupby('retweet_tid')
    # for each retweet id group counting #of unique tid
    ranks_df['tweet_share_count'] = grouped_tweets['tid'].transform('nunique')
    # get min time in posted timestamp for each grp and assign it as first_share_date
    ranks_df['first_share_date'] = grouped_tweets['postedtime'].transform('min')
    # Ranking timestamp for each group as they appear
    ranks_df['rank'] = grouped_tweets['postedtime'].rank(ascending=True, method='first')
    # calculating percentile of shares
    ranks_df['perc_shares'] = ranks_df['rank'] / ranks_df['tweet_share_count']

    ranks_df['seconds_from_1st_share'] = (ranks_df['postedtime'] - ranks_df['first_share_date']).dt.total_seconds()
    ranks_df = ranks_df.sort_values(by='retweet_tid')
    # find retweet's with an unusual fast second share and keep the quickest
    filter_ranks_df = ranks_df[ranks_df['rank'] == 2].copy(deep=True)
    filter_ranks_df['seconds_from_1st_share'] = filter_ranks_df.groupby('retweet_tid')[
        'seconds_from_1st_share'].transform('min')
    filter_ranks_df = filter_ranks_df[['retweet_tid', 'seconds_from_1st_share']]
    filter_ranks_df = filter_ranks_df.drop_duplicates()
    filter_ranks_df = filter_ranks_df[
        filter_ranks_df['seconds_from_1st_share'] <= filter_ranks_df['seconds_from_1st_share'].quantile(0.1)]

    # filter ranks_df that join with filter_ranks_df
    ranks_df = ranks_df[ranks_df.set_index('retweet_tid').index.isin(filter_ranks_df.set_index('retweet_tid').index)]
    # filter values by 0.5
    ranks_sub_df = ranks_df[ranks_df['perc_shares'] > 0.5].copy(deep=True)
    ranks_sub_df['seconds_from_1st_share'] = ranks_sub_df.groupby('retweet_tid')['seconds_from_1st_share'].transform(
        'min')
    ranks_sub_df = ranks_sub_df[['retweet_tid', 'seconds_from_1st_share']]
    ranks_sub_df = ranks_sub_df.drop_duplicates()

    summary_secs = ranks_sub_df['seconds_from_1st_share'].describe()
    coordination_interval = ranks_sub_df['seconds_from_1st_share'].quantile(0.1)
    coord_interval = (None, None)
    if coordination_interval == 0:
        coordination_interval = 1
        coord_interval = (summary_secs, coordination_interval)
    else:
        coord_interval = (summary_secs, coordination_interval)

    return coord_interval


def coord_shares(datarows):
    coordination_interval = get_estimated_threshold(datarows)
    coordination_interval = coordination_interval[1]

    orig_df = pd.DataFrame(datarows['retweet_tid'].value_counts())
    orig_df.reset_index(level=0, inplace=True)
    orig_df.columns = ['retweet_tid', "share_count"]
    orig_df = orig_df[orig_df["share_count"] > 1]
    orig_df = orig_df.sort_values('retweet_tid')
    shares_df = datarows[datarows.set_index('retweet_tid').index.isin(orig_df.set_index('retweet_tid').index)]

    data_list = []
    retweets_count = orig_df.shape[0]
    i = 0

    for index, row in orig_df.iterrows():
        i = i + 1
        try:
            print(f"processing {i} of {retweets_count}, retweet_tid={row['retweet_tid']}")
            summary_df = shares_df[shares_df['retweet_tid'] == row['retweet_tid']].copy(deep=True)
            summary_df['postedtime'] = summary_df['postedtime'].astype('datetime64[ns]')
            date_series = summary_df['postedtime'].astype('int64') // 10 ** 9
            max_value = date_series.max()
            min_value = date_series.min()
            div = (max_value - min_value) / coordination_interval + 1
            summary_df["cut"] = pd.cut(summary_df['postedtime'], int(div)).apply(lambda x: x.left).astype(
                'datetime64[ns]')
            cut_gb = summary_df.groupby('cut')
            summary_df.loc[:, 'count'] = cut_gb['cut'].transform('count')
            summary_df.loc[:, 'retweet_tid'] = row['retweet_tid']
            summary_df.loc[:, 'share_date'] = cut_gb['postedtime'].transform(lambda x: [x.tolist()] * len(x))
            summary_df = summary_df[['cut', 'count', 'share_date', 'retweet_tid', 'screen_name_from']]
            summary_df = summary_df[summary_df['count'] > 1]
            if summary_df.shape[0] > 1:
                summary_df = summary_df.loc[summary_df.astype(str).drop_duplicates().index]
                data_list.append(summary_df)
        except Exception as e:
            pass

    data_df = pd.concat(data_list)

    coor_shares_df = data_df.reset_index(drop=True).apply(pd.Series.explode).reset_index(drop=True)
    shares_df = shares_df.reset_index(drop=True)
    shares_df.loc[:, 'coord_expanded'] = shares_df['retweet_tid'].isin(coor_shares_df['retweet_tid'])
    shares_df.loc[:, 'coord_date'] = shares_df['postedtime'].astype('datetime64[ns]').isin(coor_shares_df['share_date'])

    shares_df.loc[:, 'is_coordinated'] = shares_df.apply(
        lambda x: True if (x['coord_expanded'] and x['coord_date']) else False, axis=1)
    shares_df.drop(['coord_expanded', 'coord_date'], inplace=True, axis=1)

    highly_connected_graph, q = build_graph(shares_df, coor_shares_df)
    return shares_df, highly_connected_graph, q


if __name__ == '__main__':
    filepath = 'SWM-dataset.csv'
    datarows = pd.read_csv(filepath)
    datarows = datarows.drop_duplicates()
    datarows['year'] = datarows['postedtime'].astype('datetime64[ns]').dt.year
    #call 2018 and 2016 and Ds2 to get intermediate results threshold1 and store the outputs in files.
    # dt_2016 = datarows[datarows['year'] == 2016]
    # dt_2016.drop(['year'], axis=1)
    # dt_2018 = datarows[datarows['year'] == 2018]
    # dt_2018.drop(['year'], axis=1)
    # shares_df_2016, q_2016, h_c_2016 = coord_shares(dt_2016)
    # shares_df_2018, q_2018, h_c_2018 = coord_shares(dt_2018)