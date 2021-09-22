import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import argparse
from itertools import chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-dialog-concept", 
                        default="concepts_file/dial_mindepth{}_{}_{}.npy", 
                        type=str, required=False,
                        help="path skeleton for dialogues concepts"
                             "min_depth, filter_name, split")
    parser.add_argument("--path-desc-concept", 
                        default="concepts_file/desc_mindepth{}_{}_{}_{}.npy",
                        type=str, required=False,
                        help="path skeleton for descs concepts,"
                             "min_depth, filter_name, split, id")
    parser.add_argument("--min-wn-depth", default=4, type=int, required=False,
                        help="the minimum wordnet depth filter.")
    parser.add_argument("--filter-name", type=str, required=False,
                        default="EMNLP_filter",
                        help="the name of the filter, added in the saved filename")
    parser.add_argument("--min-dist", 
                        default=1,
                        type=int, required=False,
                        help="min wn distance")
    
    args = parser.parse_args()
    min_depth = args.min_wn_depth
    filter_name = args.filter_name
    min_distance = args.min_dist
    dial_path = args.path_dialog_concept
    desc_path = args.path_desc_concept
    
    graph_path = '/home/tfangaa/projects/Deprecate/see2017seq2seq/wn_data/wngraph/wngraph.pickle'
    G = nx.read_gpickle(graph_path).to_undirected()
    def get_graph_shortest_path(G, desc, dial):
        try:
            return len(nx.shortest_path(G, desc, dial))
        except:
            return 1e3
    shortest_path = lambda desc, dial: get_graph_shortest_path(G, desc, dial) - 2
    # 1. shouldn't be too close
    # 2. cy shouldn't be one of the hypernyms of cxs
    is_codc = lambda desc_synset, dial_synsets:\
      all(
         shortest_path(desc_synset.name(), dial_synset.name())\
            >= min_distance for dial_synset in dial_synsets) and \
      all(desc_synset not in dial_synset.hypernyms() for dial_synset in dial_synsets)

    get_CODC = lambda dial_concepts, desc_concepts:\
      [desc_concept for desc_concept in desc_concepts \
         if is_codc(wn.synset(desc_concept[1]), [wn.synset(dial_concept[1]) for dial_concept in dial_concepts]) and \
          desc_concept[0] not in [dial_concept[0] for dial_concept in dial_concepts]]
    

    for proc_type in ["test", "valid", "train"]:
    
        dial_concepts = np.load(
            dial_path.format(min_depth, filter_name, proc_type),
            allow_pickle=True)
        desc_concepts_list = [np.load(desc_path.format(min_depth, filter_name, proc_type, i),\
                                      allow_pickle=True)\
                             for i in range(5)]
        desc_concepts = [list(chain(*descs)) for descs in list(zip(*desc_concepts_list))]
        codcs = []
        for dial_concept, desc_concept in tqdm(zip(dial_concepts, desc_concepts)):
            codcs.append(get_CODC(dial_concept, desc_concept))
        
        np.save(os.path.join("concepts_file/", 
                         "_".join(["codc", "mindepth{}".format(min_depth),
                          filter_name, "mindis{}".format(min_distance), proc_type])), codcs)