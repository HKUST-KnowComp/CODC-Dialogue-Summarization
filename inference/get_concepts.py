import re
import os
import spacy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from nltk.wsd import lesk

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
      for dep in deps:
        if dep[3] in candi_dep or (dep[3] == 'ROOT' and dep[1].startswith('N')) :
          # 3. determine whether this is a plausible concept:
          if dep[0] not in filter_words and not hasNumbers(dep[0]):
            synset = lesk(sent, dep[0], 'n')
            if synset is not None and synset.max_depth() >= min_depth:
              if (dep[0], synset) not in concepts:
                concepts.append((dep[0], synset))
    
    concept_synsets.append(concepts)
  return deps, concept_synsets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-dep", action='store_true',
                        help="whether filter the nouns according the specific"
                             "dependency relations")
    parser.add_argument("--min-wn-depth", default=4, type=int, required=False,
                        help="the minimum wordnet depth filter.")
    parser.add_argument("--filter-words", type=str, required=False, nargs="+",
                        default=['top', 'animation', 'photo', 'picture', 'item', 'people', 
                                 'image','photograph','thing', 'person', 'color', 'someone', 
                                 'anyone','somebody','anybody', 'something','i', 'he', 'she', 
                                 'you', 'what','which', 'this', 'that', 'any', '-PRON-'],
                        help="words to be filtered")
    parser.add_argument("--filter-name", type=str, required=False,
                        default="EMNLP_filter",
                        help="the name of the filter, added in the saved filename")
    parser.add_argument("--candi-dep", type=str, required=False, nargs="+",
                        default=['nsubj','attr','pobj','dobj','conj'],
                        help="selected dependencies")
    parser.add_argument("--path-dialog", 
                        default="/home/tfangaa/projects/OpenNMT-py-summ/"
                        "haojie/data/dialogs/dialog.{}.5ref.txt.single", 
                        type=str, required=False,
                        help="path skeleton for dialogues")
    parser.add_argument("--path-desc", 
                        default="/home/tfangaa/projects/OpenNMT-py-summ/"
                        "haojie/data/ground/desc.{}.5ref.txt.{}", 
                        type=str, required=False,
                        help="path skeleton for dialogues")

    args = parser.parse_args()
    min_depth = args.min_wn_depth
    filter_words = args.filter_words + ["-PRON-"]
    candi_dep = args.candi_dep
    select_dep = args.select_dep
    filter_name = args.filter_name

    nlp = spacy.load("en_core_web_lg")

    serialize_concept = lambda concept_synsets : \
        [[(token, synset.name()) for token, synset in concepts]
                                           for concepts in concept_synsets]

    for proc_type in ['test', 'valid', 'train' ]:
        dial_dir = args.path_dialog.format(proc_type)

        with open(dial_dir, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            lines = [line.strip() for line in lines]
            lines = [[line.strip() for line in re.split("<q>|</q>|<a>|</a>", txt)\
                      if len(line.strip())>0] for txt in lines]
            deps, concept_synsets = get_all_dependency(lines)
            
            np.save(os.path.join("concepts_file/", 
                         "_".join(["dial", "mindepth{}".format(min_depth),
                                  filter_name, proc_type]))
                    serialize_concept(concept_synsets))
        
        for i in range(5):
            desc_dir = args.path_desc.format(proc_type, i)
            with open(desc_dir, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                lines = [[line.strip()] for line in lines]
                deps, concept_synsets = get_all_dependency(lines)
                np.save(os.path.join("concepts_file/", 
                         "_".join(["desc", "mindepth{}".format(min_depth),
                                  filter_name, proc_type, i]))
                        serialize_concept(concept_synsets))