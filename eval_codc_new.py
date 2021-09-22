import spacy
import argparse
import pickle as pkl
import numpy as np
from nltk.wsd import lesk
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from tqdm import tqdm
import os
import re
from multiprocessing import Pool
from itertools import chain
import networkx as nx

nlp = spacy.load("en_core_web_sm")
num_thread = 20

graph_path = '/home/tfangaa/projects/Deprecate/see2017seq2seq/wn_data/wngraph/wngraph.pickle'
G = nx.read_gpickle(graph_path).to_undirected()

serialize_concept = lambda concept_synsets : \
        [[(token, synset.name()) for token, synset in concepts]
                                           for concepts in concept_synsets]
deserialize_concept = lambda serialized_concept_synsets : \
        [[(token, wordnet.synset(sname)) for token, sname in concepts]
                                   for concepts in serialized_concept_synsets]



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
         if is_codc(wordnet.synset(desc_concept[1]), [wordnet.synset(dial_concept[1]) for dial_concept in dial_concepts]) and \
          desc_concept[0] not in [dial_concept[0] for dial_concept in dial_concepts]]


def get_graph_shortest_path(G, desc, dial):
    try:
        return len(nx.shortest_path(G, desc, dial))
    except:
        return 1e3
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def get_all_dependency(data):
    concept_synsets = []
    for lines in tqdm(data):
        concepts = []
        for line in lines:
            # 1. parse
            doc = nlp(line)
            sent = line.split()
            deps = [(token.lemma_, token.tag_, token.head.lemma_, token.dep_) \
                       for token in doc]
            # 2. select candidate dependency
            # lesk try
            try:
                for dep in deps:
                    if dep[3] in candi_dep or (dep[3] == 'ROOT' and dep[1].startswith('N')) :
                        # 3. determine whether this is a plausible concept:
                        if dep[0] not in filter_words and not hasNumbers(dep[0]):
                            synset = lesk(sent, dep[0], 'n')
                            if synset is not None and synset.max_depth() >= min_depth:
                                if (dep[0], synset) not in concepts:
                                    concepts.append((dep[0], synset))
            except:
                print("error at", line)
        concept_synsets.append(concepts)
    return deps, serialize_concept(concept_synsets)
  


def get_concepts_threads(num_thread, lines):
    workers = Pool(num_thread)
    all_results = []
    for i in range(num_thread):
        tmp_result = workers.apply_async(
            get_all_dependency, 
            args=(lines[i*(len(lines)//num_thread+1): (i+1)*(len(lines)//num_thread+1)],) )
        all_results.append(tmp_result)
    workers.close()
    workers.join()

    concept_synsets = list(chain(*[tmp_result.get()[1] for tmp_result in all_results]))
    return concept_synsets  
  
def get_codc_threads(num_thread, dial_concept_synsets, desc_concept_synsets_list):
    num_lines = len(dial_concept_synsets)
    workers = Pool(num_thread)
    all_results = []
    for i in range(num_thread):
        start_idx = i*(num_lines//num_thread+1)
        end_idx = (i+1)*(num_lines//num_thread+1)
        tmp_result = workers.apply_async(
            get_batch_codc, 
            args=(dial_concept_synsets[start_idx:end_idx],
                 [desc_concept_synsets_list[k][start_idx:end_idx] for k in range(len(desc_concept_synsets_list))]) )
        all_results.append(tmp_result)
    workers.close()
    workers.join()

    codcs = list(chain(*[tmp_result.get() for tmp_result in all_results]))
    return codcs
  

def get_batch_codc(batch_dial_concept_synsets, batch_desc_concept_synsets_list):
    grt_codcs = []
    for i, dial_concept in tqdm(enumerate(batch_dial_concept_synsets)):
        grt_codc = list()
        try:
            for k in range(len(batch_desc_concept_synsets_list)):
                grt_codc.append(get_CODC(dial_concept, batch_desc_concept_synsets_list[k][i]))
        except:
            print(i, dial_concept)
        grt_codcs.append(grt_codc)
    return grt_codcs
  
  
def compute_synsets_hit(grt_concept, pred_concept, method="word"):
    if method == "word":
        return grt_concept[0] == pred_concept[0]
    elif method == "synset":
        return grt_concept[1] == pred_concept[1]

def codc_score_mean(grt_codcs_list, pred_codc_list, hit_method="word"):
    hit = 0
    grt_cnt = 0
    pred_cnt = 0
    for i in tqdm(range(len(grt_codcs_list))):
        grt_codcs, pred_codc = grt_codcs_list[i], pred_codc_list[i]
        for grt_codc in grt_codcs:
            grt_cnt += len(grt_codc)
            pred_cnt += len(pred_codc)
            for pred_concept in pred_codc:
                for grt_concept in grt_codc:
                    if compute_synsets_hit(grt_concept, pred_concept, hit_method):
                        hit += 1
                        break
    prec = hit / pred_cnt
    recall = hit / grt_cnt
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1
def codc_score_max(grt_codcs_list, pred_codc_list, hit_method="word"):
    prec_nom = 0
    prec_den = 0

    recall_nom = 0
    recall_den = 0

    for i in tqdm(range(len(grt_codcs_list))):
        grt_codcs, pred_codc = grt_codcs_list[i], pred_codc_list[i]

        grt_cnts = []
        pred_cnts = []
        hit_cnts = []
        for grt_codc in grt_codcs:
            hit = 0
            grt_cnts.append(len(grt_codc))
            pred_cnts.append(len(pred_codc))
            for pred_concept in pred_codc:
                for grt_concept in grt_codc:
                    if compute_synsets_hit(grt_concept, pred_concept, hit_method):
                        hit += 1
                        break
            hit_cnts.append(hit)
        

        if 0 in pred_cnts:
            continue
        i_prec = np.argmax(np.array([h/p for h, p in zip(hit_cnts, pred_cnts)]))
        prec_nom += hit_cnts[i_prec]
        prec_den += pred_cnts[i_prec]

        if 0 in grt_cnts:
            # precision as usual, recall do nothing # 暂时这样，需要讨论
            # 如果recall donothing的话，recall=100（总有=0的东西存在）
            continue
        recall_score = lambda h, g: h/g if g !=0 else 0
        i_recall = np.argmax(np.array([recall_score(h, g) for h, g in zip(hit_cnts, grt_cnts)]))
        recall_nom += hit_cnts[i_recall]
        recall_den += grt_cnts[i_recall]
        # print(i, recall_nom, recall_den)

    prec = prec_nom / prec_den
    recall = recall_nom / recall_den
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1    

# grt_cache_path, cand_cache_path
def calc_codc_scores(dialog_path, reference, candidate, grt_cache_path, cand_cache_path, 
                     hit_method="word",
                     param_dict={"min_depth": 4,
                                "filter_name": "EMNLP_filter",
                                "min_distance": 0, 
                                "candi_dep":['nsubj','attr','pobj','dobj','conj'],}):
    min_depth = param_dict["min_depth"]
    filter_name = param_dict["filter_name"]
    min_distance = param_dict["min_distance"]
    candi_dep = param_dict["candi_dep"]

    grt_path = grt_cache_path + "_".join(["codc", "mindepth{}".format(min_depth),
                          filter_name, "mindis{}".format(min_distance)])
    dial_cache = dialog_path + "_".join(["codc", "mindepth{}".format(min_depth),
                          filter_name, "mindis{}".format(min_distance)])
    if os.path.exists(grt_path+".npy") and os.path.exists(dial_cache+".npy"):
        print("[CODC] Loading ground cache from %s..." % grt_path)
        grt_codcs = np.load(grt_path+".npy", allow_pickle=True)
        dial_concept_synsets = np.load(dial_cache+".npy", allow_pickle=True)
    else:
        print("[CODC] Extracting from Dialogues...")
        with open(dialog_path, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            lines = [line.strip() for line in lines]
            lines = [[line.strip() for line in re.split("<q>|</q>|<a>|</a>", txt)\
                      if len(line.strip())>0] for txt in lines]

            dial_concept_synsets = get_concepts_threads(num_thread, lines)
        np.save(dial_cache, dial_concept_synsets)

        print("[CODC] Extracting from Descriptions...")
        desc_concept_synsets_list = list()
        for i in range(5):
            desc_dir = reference[i]
            with open(desc_dir, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                lines = [[line.strip()] for line in lines]
    #             _, desc_concept_synsets = get_all_dependency(lines)
                desc_concept_synsets = get_concepts_threads(num_thread, lines)
                desc_concept_synsets_list.append(desc_concept_synsets)
        print("[CODC] Extracting CODC from References...")

        grt_codcs = get_codc_threads(num_thread, dial_concept_synsets, desc_concept_synsets_list)

        np.save(grt_path, grt_codcs)
    if os.path.exists(cand_cache_path+".npy"):
        print("[CODC] Loading candidate cache from %s..." % cand_cache_path)
        cand_codcs = np.load(cand_cache_path+".npy", allow_pickle=True)
    else:
        # Open dialog, reference, candidate files

        print("[CODC] Loading Candidate Files...")
        with open(candidate) as f:
            lines = [[line.strip()] for line in f]
            cand_summary_concepts = list()
            _, cand_summary_concepts = get_all_dependency(lines)

        print("[CODC] Extracting CODC from Candidate...")
        cand_codcs = list()
    #     cand_codcs = get_codc_threads(num_thread, dial_concept_synsets, [cand_summary_concepts])
        cand_codcs = get_batch_codc(dial_concept_synsets, [cand_summary_concepts])
        cand_codcs = [item[0] for item in cand_codcs]
        np.save(cand_cache_path, cand_codcs)
    print("[CODC] Scoring...")
    prec_mean, recall_mean, f1_mean = codc_score_mean(grt_codcs, cand_codcs, hit_method)
    prec_max, recall_max, f1_max = codc_score_max(grt_codcs, cand_codcs, hit_method)
    print()
    print("[MEAN BASED]")
    print("Precision: {:.2f}".format(prec_mean * 100))
    print("Recall: {:.2f}".format(recall_mean * 100))
    print("F1: {:.2f}".format(f1_mean * 100))
    print("[MAX BASED]")
    print("Precision: {:.2f}".format(prec_max * 100))
    print("Recall: {:.2f}".format(recall_max * 100))
    print("F1: {:.2f}".format(f1_max * 100))

    d = {"MEAN_CODC_p": prec_mean,
        "MEAN_CODC_r": recall_mean,
        "MEAN_CODC_f1": f1_mean,
        "MAX_CODC_p": prec_max,
        "MAX_CODC_r": recall_max,
        "MAX_CODC_f1": f1_max,
    }
    return d
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialog-path', '-d', type=str, 
                        default="haojie/data/dialogs/dialog.test.txt",
                        help='Dialogue path')
    parser.add_argument('--reference', '-r', type=str, nargs='+',
                        default=["haojie/data/ground/desc.test.5ref.txt.0",
                        "haojie/data/ground/desc.test.5ref.txt.1",
                        "haojie/data/ground/desc.test.5ref.txt.2",
                        "haojie/data/ground/desc.test.5ref.txt.3",
                        "haojie/data/ground/desc.test.5ref.txt.4",], 
                        help='Ground truth summarization directory')
    parser.add_argument('--candidate', '-c', type=str, default=None,
                        help='Candidate path')
    parser.add_argument('--use-cache', action="store_true", default=False,
                        help='Use cache generated before')
    parser.add_argument('--grt-cache-path', type=str, default="./.codc_grt_cache.pkl",
                        help='Ground truth cache path')
    parser.add_argument('--cand-cache-path', type=str, default="./.codc_cand_cache.pkl",
                        help='Prediction cache path')


    args = parser.parse_args()
    use_cache = args.use_cache
    dialog_path = args.dialog_path
    reference = args.reference
    candidate = args.candidate
    grt_cache_path = args.grt_cache_path
    cand_cache_path = candidate+"_cache"

    calc_codc_scores(dialog_path, reference, candidate, 
                   grt_cache_path, cand_cache_path, 
                       hit_method="word")
