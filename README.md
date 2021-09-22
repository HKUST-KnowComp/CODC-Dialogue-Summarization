# CODC-Dialogue-Summarization
Codes of the AKBC 2021 paper: [Do Boat and Ocean Suggest Beach? Dialogue Summarization with External Knowledge](https://openreview.net/pdf?id=AJKd0iIFMDc). 

## Data

The data can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/Eur2zirUWiRNtlusZpId39IBmXyAgl_p3YUZ-CQi8_UizQ?e=ND1lpR).

## Experiments

### 1. Inference Module

Check `inference/pipeline.ipynb` for more details.

1. Extract concepts from dialogues and summaries (`get_concepts.py`).
2. Get Concepts Out-of Dialogue Context (`get_codc.py`).
3. Build Co-occurence graph (`build_graph.py `).
4. Get the inferred knowledge (`get_features.py`).
5. Classifier (`classify.ipynb`).

### 2. Summarization Module

Prepare training data. Training dataset preparation for baselines:

```
sh scripts/prepare_training_data_baseline.sh
```

And modify the paths to the training knowledge in `prepare_training_data_knowattn.sh`, which is the `-train_know` argument.

```
sh scripts/prepare_training_data_knowattn.sh
```

Generate summaries:

```
python translate.py -gpu 0 \
     -knowledge \
     -batch_size 20 \
     -beam_size 4 \
     -model models/random_train_know/trans_copy_top5_prop0.2_step_35000.pt \
     -src codc_data/dialogs/dialog.test.txt \
     -know codc_data/inference/knowledge/rfdepth12_mindepth4_mindis0_EMNLP_filter_nofilter_test_top13.txt \
     -output decodings/prop0.2_tfdep12_top13_35k_trainedtop5 \
     -min_length 5 \
     -max_length 15
```



### 3. Evaluation

`python eval_codc.py`