python preprocess.py -knowledge \
        -train_src /home/tfangaa/projects/process_dial2desc/all_5_caps/train_dialog_shuffle \
        -train_tgt /home/tfangaa/projects/process_dial2desc/all_5_caps/train_descs_shuffle \
        -valid_src /home/data/corpora/Dial2Desc/dialog.test.txt \
        -valid_tgt /home/data/corpora/Dial2Desc/desc.test.txt \
        -save_data dataset/EMNLP2020-retro/nov_know/train_dep5_top8 \
        -src_seq_length 10000 \
        -tgt_seq_length 10000 \
        -src_seq_length_trunc 200 \
        -tgt_seq_length_trunc 15 \
        -dynamic_dict \
        -share_vocab \
        -shard_size 500000 \
        -src_vocab_size 20000 \
        -tgt_vocab_size 20000 \
        -know_vocab_size 20000 \
        -know_seq_length 50 \
        -know_seq_length_trunc 15 \
        -train_know inference/EMNLP2020/knowledge/5ref/train_depth5_top_8 \
        -valid_know /home/tfangaa/projects/process_dial2desc/depen-parse/similarity/knowledge_file/ood_rf_top5 \
        -overwrite



nohup bash scripts/ACL2021/preproc_know.sh dataset/EMNLP2020-retro/esa_v1/train_esa_top1 \
     inference/EMNLP-retro/knowledge/train-knowledge/esa-classifier/fivecap_rfdepth12_mindepth4_mindis0_EMNLP_filter_nofilter_train_top1.txt &

nohup bash scripts/ACL2021/preproc_know.sh dataset/EMNLP2020-retro/random_train_know/fivecap_top5_prop0.25_train \
     inference/EMNLP-retro/knowledge/train-knowledge/random-sample/fivecap_top5_prop0.25_train.txt  &

nohup bash scripts/ACL2021/preproc_know.sh dataset/EMNLP2020-retro/random-fivecap/fivecap_top3_prop0.4_train \
     inference/EMNLP-retro/knowledge/train-knowledge/random-fivecap/fivecap_top3_prop0.4_train.txt  &    




