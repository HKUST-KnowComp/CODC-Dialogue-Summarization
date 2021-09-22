import os
import itertools
import argparse
import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy import spatial
from tqdm import tqdm
from collections import Counter
from itertools import chain

from multiprocessing import Pool

from scipy.sparse import lil_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(229)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', '-save_path', type=str, 
                        default="attribute_file/",
                        help='path of embedding file')
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
    parser.add_argument('--codc_path', '-codc_path', type=str, 
                        default="concepts_file/codc_mindepth{}_{}_mindis{}_{}.npy",
                        help='path of codc file')
    parser.add_argument('--min_cooccurance', '-min_cooccurance', type=int, default=2,
                        help='filter when finding candidates')
    parser.add_argument('--num_thread', '-num_thread', type=int, default=20,
                        help='number of threads to process co_prob')
    
    args = parser.parse_args()
    save_path = args.save_path
    concept_dial_path = args.concept_dial_path
    min_dist = args.min_dist
    min_depth = args.min_wn_depth
    min_cooccurance = args.min_cooccurance
    max_connect_dist = args.connect_dist
    concept_filter_name = args.filter_name
    codc_path = args.codc_path
    num_thread = args.num_thread
    proc_type = "train"
    graph_filter_name = "nofilter"
    
    concept_dials = np.load(concept_dial_path.format(min_depth, concept_filter_name, proc_type), allow_pickle=True)
    codcs = np.load(codc_path.format(min_depth, concept_filter_name, min_dist, proc_type),allow_pickle=True)
    codcs_names = [[i[0] for i in item] for item in codcs]
    
    candi_names = np.load(
        "attribute_file/candi_mindepth4_codcdis1_EMNLP_filter_nofilter_train_neg10_name.npy",
        allow_pickle=True)
    sim_save_esa = os.path.join(save_path, '_'.join([
                          'esa_sim',
                          "mindepth{}".format(min_depth),
                          "codcdis{}".format(min_dist),
                          concept_filter_name, 
                          graph_filter_name,
                          proc_type,
                          "newcandidate"]))
    
    concepts_dials_nouns = [[n[0] for n in item] for item in concept_dials]
    print("processing esa")
    print(len(concepts_dials_nouns), len(candi_names))
    
    esa_sims = get_esa_similarities(num_thread, 
                        concepts_dials_nouns[40000:], 
                        candi_names[40000:])
    np.save(sim_save_esa + "40000_end", esa_sims)