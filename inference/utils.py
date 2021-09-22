# prepare X and ys
from tqdm import tqdm
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
def rf_clf(X, y, X_test=None, y_test=None, depth=10, n_est=10, class_weight=dict({0:1, 1:1})):
    clf_rf = RandomForestClassifier(
      max_depth=depth, random_state=0, n_estimators=n_est, class_weight=class_weight)
    clf_rf.fit(X, y)

    print("acc:", clf_rf.score(X_test, y_test))
    if X_test is not None:
        pred_rf = clf_rf.predict(X_test)
        recall = recall_score(y_test, pred_rf, average='macro')
        f1 = f1_score(y_test, pred_rf, average='macro')
        print("recall:", recall, "f1:", f1)
    return clf_rf

def prepare_dataset(G_attrs, codcs_name, 
                    keys=['weight', 'wup', 'path', 'lch', 
                          'shortest_path_wn', 'w2v', 'glove', 'esa']):
    """
      return: 
        X attributes
        y denoting whether it is a codc
    """
    X = []
    y = []
    for attrs, codc in tqdm(zip(G_attrs, codcs_name)):
      for word in attrs:
        X.append([attrs[word][key] for key in keys])
        y.append(int(word in codc))
    return X, y
def get_f1(preds, refs):
  p_list, r_list, f1_list = [], [], []
  for pred, ref in zip(preds, refs):
    p, r, f1 = metric(pred, ref)
    p_list.append(p)
    r_list.append(r)
    f1_list.append(f1)
  print('number of zeros hits', len(np.where(np.array(r_list) == 0)[0]))
  return np.mean(p_list), np.mean(r_list), np.mean(f1_list)

def metric(pred, ref):
  TP=len(set(ref) & set(pred))
  TN=len(set(pred)-set(ref))
  FP=len(set(ref)-set(pred))
  precision = TP / (TP + TN) if TP > 0 else 0
  recall = TP / (TP + FP) if TP > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if TP > 0 else 0
  return precision, recall, f1

def write_to_file(retrieved, filename):
  with open(filename, 'w') as writer:
    for line in retrieved:
      writer.write(' '.join(line)+'\n')
  return
def eval_ensemble(clf, G_attrs, codcs, 
                  keys=['weight', 'wup', 'path', 'lch', 'shortest_path_wn', 'w2v', 'glove', 'co_prob']):
  # calculate scores for every key word
  # check f1 @ 10, @ 20, @30, @50, @100.
  ens_keys = []
  ens_values = []
  for attrs in tqdm(G_attrs):
    # prepare to dataset
    X = np.zeros([len(attrs), len(keys)])
    candi_keys = ['' for i in range(len(attrs))]
    for i, word in enumerate(attrs):
      X[i] = [attrs[word][key] for key in keys]
      candi_keys[i] = word
    if len(attrs) == 0:
      ens_keys.append([])
      ens_values.append([])
      continue
    values = clf.predict_proba(X)[:, 1]
    index = np.argsort(values)[::-1]
    candi_keys = np.array(candi_keys)[index]
    values = values[index]
    ens_keys.append(candi_keys)
    ens_values.append(values)
  # evaluate
  for top_k in [1, 2, 3, 4, 5, 10]:
    retrieved = []
    for k, v in zip(ens_keys, ens_values):
      retrieved.append(k[:min(len(k), top_k)])
    print('topk', top_k, get_f1(retrieved, codcs))
  return ens_keys, ens_values 

def eval_ensemble_pure_clf(clf, G_attrs, codcs, keys=['weight', 'wup', 'path', 'lch', 'shortest_path_wn', 'w2v', 'glove', 'co_prob']):
  retrieved = []
  for attrs in tqdm(G_attrs):
    X = np.zeros([len(attrs), len(keys)])
    candi_keys = ['' for i in range(len(attrs))]
    for i, word in enumerate(attrs):
      X[i] = [attrs[word][key] for key in keys]
      candi_keys[i] = word
    # calc scores
    if len(attrs) == 0:
      continue
    values = clf.predict(np.array(X))
    retrieved.append(np.array(candi_keys)[values==1])
  return retrieved