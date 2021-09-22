import argparse
import networkx as nx
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn

MAX_PATH_LENGTH=20 + 2
synset_cache = {}
path_sim_cache = {}
wup_sim_cache = {}
lch_sim_cache = {}
shortest_path_cache = {}
def get_synset(concept):
    # convert string to wn.synset
    if concept in synset_cache:
        return synset_cache[concept]
    else:
        synset_cache[concept] = wn.synset(concept[1])
        return synset_cache[concept]

def get_path_sim(dial, codc):
    if (dial, codc) not in path_sim_cache:
        path_sim = get_synset(dial).path_similarity(get_synset(codc))
        path_sim_cache[(dial, codc)] = path_sim
        path_sim_cache[(codc, dial)] = path_sim
        return path_sim_cache[(codc, dial)]
    else:
        return path_sim_cache[(codc, dial)]

def get_wup_sim(dial, codc):
    if (dial, codc) not in wup_sim_cache:
        wup_sim = get_synset(dial).wup_similarity(get_synset(codc))
        wup_sim_cache[(dial, codc)] = wup_sim
        wup_sim_cache[(codc, dial)] = wup_sim
        return wup_sim_cache[(codc, dial)]
    else:
        return wup_sim_cache[(codc, dial)]

def get_lch_sim(dial, codc):
    if (dial, codc) not in lch_sim_cache:
        lch_sim = get_synset(dial).lch_similarity(get_synset(codc))
        lch_sim_cache[(dial, codc)] = lch_sim
        lch_sim_cache[(codc, dial)] = lch_sim
        return lch_sim_cache[(codc, dial)]
    else:
        return lch_sim_cache[(codc, dial)]



def get_graph_shortest_path(G, desc, dial):
    if (desc, dial) in shortest_path_cache:
        return shortest_path_cache[(desc, dial)]
    else:
        try:
            shortest_path_cache[(desc, dial)] = \
                min(MAX_PATH_LENGTH, len(nx.shortest_path(G, desc, dial)))
            shortest_path_cache[(dial, desc)] = \
                min(MAX_PATH_LENGTH, shortest_path_cache[(desc, dial)])
            return shortest_path_cache[(desc, dial)]
        except:
            shortest_path_cache[(desc, dial)] = MAX_PATH_LENGTH
            shortest_path_cache[(dial, desc)] = MAX_PATH_LENGTH
            return shortest_path_cache[(desc, dial)]

shortest_path = lambda desc, dial: max(0, get_graph_shortest_path(G_wn, desc, dial) - 2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-dialog-concept", 
                        default="./concepts_file/dial_mindepth{}_{}_train.npy", 
                        type=str, required=False,
                        help="path for training dialogues concepts."
                                "arg:wn_depth and filter_name")
    parser.add_argument("--path-codc", 
                        default="./concepts_file/codc_mindepth{}_{}_mindis{}_train.npy",
                        type=str, required=False,
                        help="path for training codc"
                               "args:wn_depth, filter_name, min_dis")
    parser.add_argument("--min-depth", 
                        default=4,
                        type=int, required=False,
                        help="minimum depth of WordNet concept")
    parser.add_argument("--min-dis", 
                        default=0,
                        type=int, required=False,
                        help="minimum wordnet graph distance to be a CODC")
    parser.add_argument("--filter-name", type=str, required=False,
                        default="EMNLP_filter",
                        help="the name of the filter, added in the saved filename")
    parser.add_argument("--connect-dist", 
                        default=4,
                        type=int, required=False,
                        help="connect two concepts in the graph if their "
                             "hops in wordnet is less than connect_dist")
    parser.add_argument("--filter-weight", action='store_true',
                        help="whether to filter out the edges that occurs less than once")

    args = parser.parse_args()
    
    graph_path = '/home/tfangaa/projects/Deprecate/see2017seq2seq/wn_data/wngraph/wngraph.pickle'
    G_wn = nx.read_gpickle(graph_path).to_undirected()
    
    
    connect_dist = args.connect_dist    
    filter_weight = args.filter_weight
    concept_filter_name = args.filter_name
    
    concept_dials = np.load(
        args.path_dialog_concept.format(args.min_depth, concept_filter_name), 
        allow_pickle=True)
    codcs = np.load(
        args.path_codc.format(args.min_depth, concept_filter_name, args.min_dis),
        allow_pickle=True)

    
    G_new = nx.DiGraph()

    for dials, codcs in tqdm(zip(concept_dials, codcs)):
        # if the dial -> codc path number is less than connect_dist, then connect them
        for dial in dials:
            for codc in codcs:
                if shortest_path(dial[1], codc[1]) <= connect_dist:
                    if not G_new.has_edge(dial[1], codc):
                        G_new.add_edge(dial[1], codc, weight=1)
                        G_new[dial[1]][codc]['wup'] = get_wup_sim(dial, codc)
                        G_new[dial[1]][codc]['path'] = get_path_sim(dial, codc)
                        G_new[dial[1]][codc]['lch'] = get_lch_sim(dial, codc)
                        G_new[dial[1]][codc]['shortest_path_wn'] = -shortest_path(dial[1], codc[1]) 
                        # here we take the negative of shortest_path, 
                        # as we require the minimum shortest path
                    else:
                        G_new[dial[1]][codc]['weight'] += 1

    if filter_weight:
        # normalize weight (with filter)
        for node in G_new.nodes():
            total_weights = sum([G_new[node][n]['weight'] \
                                 for n in nx.neighbors(G_new, node)\
                                 if G_new[node][n]['weight'] > 1])
            rm_list = []
            for n in nx.neighbors(G_new, node):
                if G_new[node][n]['weight'] == 1:
                    rm_list.append(n)
                else:
                    G_new[node][n]['weight'] /= total_weights
            for n in rm_list:
                G_new.remove_edge(node, n)
        nx.write_gpickle(G_new, "concepts_file/codc_graph/G_mindepth{}_codcdis{}_{}_maxconndist{}_weightfilter.pickle"\
                         .format(args.min_depth, args.min_dis, 
                                 concept_filter_name, args.connect_dist))
    else:
        # normalize weight (no filter)
        for node in G_new.nodes():
            total_weights = sum([G_new[node][n]['weight'] for n in nx.neighbors(G_new, node)])
            for n in nx.neighbors(G_new, node):
                G_new[node][n]['weight'] /= total_weights
        nx.write_gpickle(G_new, "concepts_file/codc_graph/G_mindepth{}_codcdis{}_{}_maxconndist{}_nofilter.pickle"\
                        .format(args.min_depth, args.min_dis, 
                                concept_filter_name, args.connect_dist))