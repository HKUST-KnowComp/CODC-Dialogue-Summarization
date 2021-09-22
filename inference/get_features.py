"""
    1. prepare candidates
    2. Calculate w2v/glove similarity and other features.
    3. prepare training and testing dataset
"""

import os
import gensim
import itertools
import argparse
import networkx as nx
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from scipy import spatial
from tqdm import tqdm
from collections import Counter
from itertools import chain

from multiprocessing import Pool

from scipy.sparse import lil_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(229)

cos_sim = lambda x, y : 1 - spatial.distance.cosine(x, y)
cos_sim_esa = lambda x, y : cosine_similarity(vstack([x, y]))[0][1]

def calc_sim_esa(concepts_dial, all_candis, emb_dict):
    # Given the candidates, return the value of candidate retrieve knowledge
    res = []
    dim = list(emb_dict.values())[0].shape[1]
    for dial_nouns, candis in tqdm(zip(concepts_dial, all_candis)):
        # calculate the average of dial_nouns vectors
        dial_vec = [emb_dict[w] for w in dial_nouns if w in emb_dict]
        if len(dial_vec) > 0:
            dial_vec_mean = sum(dial_vec) / len(dial_vec)
        else:
            dial_vec_mean = lil_matrix((1, dim))
        res_dict = dict([(w, cos_sim_esa(emb_dict[w], dial_vec_mean)) for w in candis if w in emb_dict])
        res.append(res_dict)
    return res

def calc_sim(concepts_dial, all_candis, emb_dict):
    # Given the candidates, return the value of candidate retrieve knowledge
    res = []
    for dial_nouns, candis in tqdm(zip(concepts_dial, all_candis)):
        # calculate the average of dial_nouns vectors
        dial_vec_mean = sum([emb_dict[w] for w in dial_nouns if w in emb_dict ]) / len(dial_nouns)
        res_dict = dict([(w, cos_sim(emb_dict[w], dial_vec_mean)) for w in candis if w in emb_dict])
        res.append(res_dict)
    return res

def get_similarities(dial_nouns, candi_nouns, model):
    """
      dial_nouns: (list) of list of nouns
      candi_nouns: (list) of list of nouns
      emb_path: (str) paath to word embedding
      binary: (boolean) whether this is binary file
    """
    # 1. get vocabulary and see how many of the vocabulary is covered
    dial_vocab = np.unique(list(itertools.chain.from_iterable(dial_nouns)))
    candi_vocab = np.unique(list(itertools.chain.from_iterable(candi_nouns)))
    all_vocab = np.unique(np.append(dial_vocab, candi_vocab))
    
    emb_dict = dict([(w, model[w]) for w in all_vocab if w in model])
    no_emb_list = [w for w in all_vocab if w not in model]
    print('num of vocab', len(all_vocab), '\nnumber of no_emb', len(no_emb_list), '\n')
    
    # 2. get similarities
    similarities = calc_sim(dial_nouns, candi_nouns, emb_dict)
    return similarities

def get_co_prob(graph, concepts_dials, candidates):
    """
        get cooccurance probability of candidates
    """
    props = []
    for dial, candis in tqdm(zip(concepts_dials, candidates)):
        prop = {}
        for i in range(len(dial)):
            for j in range(i + 1, len(dial)):
                for candi in candis:
                    prop[candi[0]] = 0 # initialized as 0 ( if no co-occurance, then 0.) 
                    try:
                        if candi in graph.neighbors(dial[i][1]) and candi in graph.neighbors(dial[j][1]):
                            p = graph[dial[i][1]][candi]['weight'] * graph[dial[j][1]][candi]['weight']
                            if p > prop[candi[0]]:
                                prop[candi[0]] = p
                    except:
                        pass
        props.append(prop)

    return props    

def get_common_neighbors(G, concept_dials, min_cooccurance=2):
    # 
    ret_knows = []
    for dials in tqdm(concept_dials):
        ret_know = []
        for dial in dials:
            try:
                ret_know += list(G.neighbors(dial[1]))
            except:
                pass
        know_counter = Counter(ret_know)
        ret_knows.append([key for key in know_counter if know_counter[key] >= min_cooccurance])
    return ret_knows

def update_attr(attr_dict, new_dict, keys=['weight', 'wup', 'path', 'lch', 'shortest_path_wn',]):
    """
      update the attributes dict
      shortest_path (negative)
      Keep the largest feature.
    """
    for key in keys:
        if attr_dict.get(key, -1e5) < new_dict.get(key, 1e-5):
            attr_dict[key] = new_dict[key]
    return attr_dict

def prepare_dataset(G_attrs, codcs_name, 
    keys=['weight', 'wup', 'path', 'lch', 'shortest_path_wn', 'w2v', 'glove', 'co_prob']):
    """
      return: 
        X attributes
        y denoting whether it is a codc
    """
    num_item = sum([len(list(G_attrs[i].keys())) for i in range(len(G_attrs))])
    X = np.zeros([num_item, len(keys)])
    y = np.zeros(num_item)
    i = 0
    for attrs, codc in tqdm(zip(G_attrs, codcs_name)):
        for word in attrs:
            X[i] = [attrs[word][key] for key in keys]
            y[i] = int(word in codc)
            i += 1
    return X, y

def build_esa_embedding(lines, all_vocab, dim):
  esa_dict = {}
  for line in lines:
    w, _, attr = line.strip().split()
    if not w in all_vocab:
      continue
    # only deal with vocab words
    vec = lil_matrix((1, dim))
    attr = np.array([item.split(',') for item in attr.split(';')[:-1]])
    keys = [int(item[0]) for item in attr]
    values = [float(item[1]) for item in attr]
    vec[0, keys] = values
    esa_dict[w] = vec
  return esa_dict
def get_esa_dim(esa_file):
    dim = 0
    for line in esa_file:
      for item in line.strip().split()[2].split(';')[:-1]:
        item = item.split(',')
        if int(item[0]) > dim:
          dim = int(item[0])
    return dim

def get_esa_similarities(num_thread, concepts_dials_nouns, candidates_nouns):
    dial_vocab = np.unique(list(itertools.chain.from_iterable(concepts_dials_nouns)))
    candi_vocab = np.unique(list(itertools.chain.from_iterable(candidates_nouns)))
    all_vocab = np.unique(np.append(dial_vocab, candi_vocab))
    if os.path.exists('mesa_emb_' + proc_type + '.npy'):
        mesa_emb = np.load('mesa_emb_' + proc_type + '.npy', allow_pickle=True)[()]
    else:
        esa_file = open('/home/data/corpora/wikipedia/backup_uiuc_Web/MemoryBasedESA.txt', 'r').readlines()
        dim = get_esa_dim(esa_file)
        mesa_emb = build_esa_embedding(esa_file, all_vocab, dim)
        np.save('mesa_emb_' + proc_type, mesa_emb)
        print('finish building mesa embedding')
    
    workers = Pool(num_thread)
    all_results = []
    for i in range(num_thread):
        tmp_result = workers.apply_async(
          calc_sim_esa, #(concepts_dial, all_candis, emb_dict), 
          args=( 
                concepts_dials_nouns[i* (len(concepts_dials_nouns)//num_thread+1):\
                              (i+1)*(len(concepts_dials_nouns)//num_thread+1)], 
                candidates_nouns[i* (len(concepts_dials_nouns)//num_thread+1):\
                              (i+1)*(len(concepts_dials_nouns)//num_thread+1)],
                mesa_emb)
            )
        all_results.append(tmp_result)

    workers.close()
    workers.join()

    all_results = [tmp_result.get() for tmp_result in all_results]
    esa_sims = list(chain(*all_results))
    return esa_sims   


def select_for_training(codc_candis, codcs_names, prop=10):
    codc_candis_names = [[n[0] for n in item] for item in codc_candis]
    print("selecting candidate codcs for training set...")
    # prop: the proportion of negative samples
    Y = []
    pos_indexes = [] # reindex the indexes for positive examples
    neg_indexes = [] # neg
    idx = 0
    for candis, codcs in tqdm(zip(codc_candis_names, codcs_names)):
        Y.append([c in codcs for c in candis])
        pos_indexes.append([i for i in range(len(candis)) if Y[-1][i]==1 ])
        neg_indexes.extend([i + idx for i in range(len(candis)) if Y[-1][i]==0 ])
        idx += len(candis)
    
    assert sum(list(chain(*Y))) == len(list(chain(*pos_indexes)))
    
    num_pos =  sum(list(chain(*Y)))
    num_neg = prop * num_pos
    print("number of positive samples:", num_pos)
    print("number of sampled negative samples:", num_neg)
    selected_neg_idx = np.random.choice(neg_indexes, num_neg, replace=False)
    selected_neg_idx = np.sort(selected_neg_idx)
    
    selected_codc_candis = []
    start_idx = 0
    end_idx = 0
    sid = 0
    for i, (candis, codcs) in tqdm(enumerate(zip(codc_candis, codcs_names))):
        end_idx += len(candis)
        pos_idx = pos_indexes[i]
        tmp_candis = []
        while sid < len(selected_neg_idx) and selected_neg_idx[sid] < end_idx:
            tmp_candis.append(candis[selected_neg_idx[sid]-start_idx])
            sid += 1
        tmp_candis.extend([candis[idx] for idx in pos_idx])
#         kept_idx_neg = selected_neg_idx[
#             (selected_neg_idx>=start_idx) & (selected_neg_idx<end_idx)
#             ]
#         selected_codc_candis.append(
#           [candis[idx] for idx in list(kept_idx_neg - start_idx) + pos_idx ]
#         )
        selected_codc_candis.append(tmp_candis)
        start_idx = end_idx
    print("avg number of neighbors", 
          sum([len(item) for item in selected_codc_candis])/len(selected_codc_candis))
    return selected_codc_candis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalculate', action='store_true')
    parser.add_argument('--save_path', '-save_path', type=str, 
                        default="attribute_file/",
                        help='path of embedding file')
    parser.add_argument('--w2v_path', '-w2v_path', type=str, 
                        default="/home/data/corpora/word_embeddings/GoogleNews-vectors-negative300.bin", 
                        help='path of w2v_path file')
    parser.add_argument('--glove_path', '-glove_path', type=str,  default="/home/data/corpora/word_embeddings/english_embeddings/glove/glove.840B.300d.w2v_format.txt", 
                        help='path of w2v_path file')
    parser.add_argument('--graph_path', '-graph_path', type=str, 
                        default="concepts_file/codc_graph/"
                        "G_mindepth{}_codcdis{}_{}_maxconndist{}_{}.pickle",
                        help="path of graph file."
                             "args: min_depth, min_dis, filter_name, "
                             "max_connection_dist, filter_graph_weight")
    parser.add_argument('--concept_dial_path', '-concept_dial_path', type=str, 
                        default="concepts_file/dial_mindepth{}_{}_{}.npy",
                        help="path of concept_dial file"
                             "args: min_depth, concept_filter_name, proc_type")
    parser.add_argument("--min-dist", 
                        default=0,
                        type=int, required=False,
                        help="min wn distance")
    parser.add_argument("--min-wn-depth", default=4, type=int, required=False,
                        help="the minimum wordnet depth filter.")
    parser.add_argument("--connect-dist", 
                        default=4,
                        type=int, required=False,
                        help="connect two concepts in the graph if their "
                             "hops in wordnet is less than connect_dist")
    parser.add_argument("--filter-name", type=str, required=False,
                        default="EMNLP_filter",
                        help="the name of the filter, added in the saved filename")
    parser.add_argument("--filter-weight", action='store_true',
                        help="whether to filter out the edges that occurs less than once")
    parser.add_argument('--codc_path', '-codc_path', type=str, 
                        default="concepts_file/codc_mindepth{}_{}_mindis{}_{}.npy",
                        help='path of codc file')
    parser.add_argument('--min_cooccurance', '-min_cooccurance', type=int, default=2,
                        help='filter when finding candidates')
    parser.add_argument('--num_thread', '-num_thread', type=int, default=20,
                        help='number of threads to process co_prob')
    parser.add_argument('--neg_prop', '-neg_prop', type=int, default=10,
                        help='proportion of randomly sampled negative samples in the training set.')
    parser.add_argument("--proc_types", type=str, required=False, nargs="+",
                        default=["test", "train"],
                        help="proc types")
    
    
    args = parser.parse_args()
    recalculate = args.recalculate
    w2v_path = args.w2v_path
    glove_path = args.glove_path
    save_path = args.save_path
    graph_path = args.graph_path
    concept_dial_path = args.concept_dial_path
    min_dist = args.min_dist
    min_depth = args.min_wn_depth
    min_cooccurance = args.min_cooccurance
    max_connect_dist = args.connect_dist
    concept_filter_name = args.filter_name
    graph_filter_weight = args.filter_weight
    codc_path = args.codc_path
    num_thread = args.num_thread
    neg_prop = args.neg_prop
    
    if graph_filter_weight:
        graph_filter_name = "weightfilter"
    else:
        graph_filter_name = "nofilter"
    
    
    G = nx.read_gpickle(graph_path.format(min_depth, min_dist, 
                concept_filter_name, max_connect_dist, graph_filter_name))
    
    for proc_type in args.proc_types:# ['test', 'train']: # 'valid']:
        # 1. Given graph, prepare training set and testing set
        candi_save_dir = os.path.join(save_path, 
                          '_'.join([
                          "candi",
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type]))
        if proc_type == "train":
            candi_save_dir = candi_save_dir+"_negprop{}".format(neg_prop)
        
        concept_dials = np.load(concept_dial_path.format(min_depth, concept_filter_name, proc_type), allow_pickle=True)
        codcs = np.load(codc_path.format(min_depth, concept_filter_name, min_dist, proc_type),allow_pickle=True)
        codcs_names = [[i[0] for i in item] for item in codcs]
        
        print('Retrieving and saving candidates')
        if os.path.exists(candi_save_dir+".npy") and not recalculate:
            codc_candis = np.load(candi_save_dir+".npy", allow_pickle=True)
        else:
            codc_candis = get_common_neighbors(G, concept_dials, min_cooccurance)
            if proc_type == "train":
                codc_candis = select_for_training(codc_candis, codcs_names, prop=neg_prop)
            np.save(candi_save_dir, codc_candis)

        # 2. get similarities between candis and its corresponding dialogs
        sim_save_esa = os.path.join(save_path, '_'.join([
                          'esa_sim',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type ]))
        
        sim_save_w2v = os.path.join(save_path, '_'.join([
                          'w2v_sim',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type ]))
        sim_save_glove = os.path.join(save_path, '_'.join([
                          'glove_sim',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type ]))
        co_occur_prob_path = os.path.join(save_path, '_'.join([
                          'co_occur_prob',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type ]))
        
        concepts_dials_nouns = [[n[0] for n in item] for item in concept_dials]
        candidates_nouns = [[n[0] for n in item] for item in codc_candis]
        
        print('processing cooccurance')
        if os.path.exists(co_occur_prob_path + '.npy') and not recalculate:
            print("...loading from", co_occur_prob_path + '.npy')
            co_probs = np.load(co_occur_prob_path+'.npy', allow_pickle=True)
        else:
            num_thread = args.num_thread
            workers = Pool(num_thread)
            all_results = []
            for i in range(num_thread):
                tmp_result = workers.apply_async(
                  get_co_prob, 
                  args=(G, 
                        concept_dials[i* (len(concept_dials)//num_thread+1):\
                              (i+1)*(len(concept_dials)//num_thread+1)], 
                codc_candis[i* (len(concept_dials)//num_thread+1):\
                              (i+1)*(len(concept_dials)//num_thread+1)]))
                all_results.append(tmp_result)

            workers.close()
            workers.join()

            all_results = [tmp_result.get() for tmp_result in all_results]
            co_probs = list(chain(*all_results))
            np.save(co_occur_prob_path, co_probs)
        
        print('processing w2v')
        if os.path.exists(sim_save_w2v+'.npy') and not recalculate:
            print("...loading from", sim_save_w2v + '.npy')
            w2v_sims = np.load(sim_save_w2v+'.npy', allow_pickle=True)
        else:
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            w2v_sims = get_similarities(concepts_dials_nouns, candidates_nouns, w2v_model)
            np.save(sim_save_w2v, w2v_sims)
        
        print('processing glove')
        if os.path.exists(sim_save_glove+'.npy') and not recalculate:
            print("...loading from", sim_save_glove + '.npy')
            glove_sims = np.load(sim_save_glove+'.npy', allow_pickle=True)
        else:
            glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False)
            glove_sims = get_similarities(concepts_dials_nouns, candidates_nouns, glove_model)
            np.save(sim_save_glove, glove_sims)
        
        print('processing esa')
        if os.path.exists(sim_save_esa+'.npy') and not recalculate:
            print("...loading from", sim_save_esa)
            esa_sims = np.load(sim_save_esa + '.npy', allow_pickle=True)
        else:
            esa_sims = get_esa_similarities(num_thread, 
                        concepts_dials_nouns, 
                        candidates_nouns)
            np.save(sim_save_esa, esa_sims)
        
        # 3. ensamble similarities and graph attributes
        
        attrs_save_dir = os.path.join(save_path, '_'.join([
                          'attrs',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type
                          ]))

        # Traverse concept_dials and candi_concepts
        G_attrs = []
        print('gathering graph attributes')
        for dials, candis, w2v, glove, co_prob, esa in tqdm(zip(concept_dials, codc_candis, w2v_sims, glove_sims, co_probs, esa_sims)):
            attr = {}
            for candi in candis:
                # check the values of different G_wn attributes
                for dial in dials:
                    if G.has_edge(dial[1], candi):
                        attr[candi[0]] = update_attr(attr.get(candi[0], {}), G[dial[1]][candi])
            # update with w2v and glove sims
            for key in attr:
                attr[key]['w2v'] = w2v.get(key, -2) # 233
                attr[key]['glove'] = glove.get(key, -2) # 233
                attr[key]['esa'] = esa.get(key, -2) # 233
                attr[key]['co_prob'] = co_prob.get(key, 0) # 0
            G_attrs.append(attr)
        np.save(attrs_save_dir, G_attrs)