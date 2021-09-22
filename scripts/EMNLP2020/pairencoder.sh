#!/usr/bin/env bash
export model_name=
export DATA= # $2
export GPU=3 # $3
export ENC=  # $4
export CUDA_VISIBLE_DEVICES=$GPU
# mkdir cache/$model_name
mkdir logs/$model_name

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
                   -pairenc \
                   -save_model models/pairenc/1ref \
                   -data dataset/pairenc/1ref \
                   -layers 4 \
                   -rnn_size 256 \
                   -word_vec_size 256 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type qa3 \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 100000 \
                   -valid_steps 5000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0  > info/pairenc/1ref &

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
                   -pairenc \
                   -train_from models/pairenc/5ref_step_100000.pt \
                   -save_model models/pairenc/5ref_from_100k \
                   -data dataset/pairenc/5ref \
                   -layers 4 \
                   -rnn_size 256 \
                   -word_vec_size 256 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type qa3 \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 200000 \
                   -valid_steps 5000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 > info/pairenc/5ref_from10k &

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
                   -pairenc \
                   -save_model models/pairenc/know_5ref_top3_r54_f37_from10k \
                   -data dataset/pairenc/5ref_top3_r54_f37 \
                   -know_train_from models/pairenc/5ref_step_100000.pt \
                   -know_train_from_type finetune \
                   -p_kgen_func mlp \
                   -knowledge \
                   -know_emb_init from_tgt \
                   -know_loss_lambda 0.0 \
                   -know_attn_type mlp \
                   -know_query ori_query \
                   -prob_logits_type include_k_context \
                   -trans_know_query_type iter_query \
                   -layers 4 \
                   -rnn_size 256 \
                   -word_vec_size 256 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type qa3 \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 100000 \
                   -valid_steps 5000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 > info/pairenc/5ref_know_from100k &

                   # -know_train_from models/pairenc/xxxxx.pt \
                   # -know_train_from_type finetune \

    
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 nohup python -u train.py \
                   -pairenc \
                   -train_from models/pairenc/know_5ref_top3_r54_f37_from10k_step_40000.pt \
                   -save_model models/pairenc/know_5ref_from40kknowref \
                   -data dataset/pairenc/5ref_top3_r54_f37 \
                   -know_train_from_type finetune \
                   -p_kgen_func mlp \
                   -knowledge \
                   -know_emb_init from_tgt \
                   -know_loss_lambda 0.0 \
                   -know_attn_type mlp \
                   -know_query ori_query \
                   -prob_logits_type include_k_context \
                   -trans_know_query_type iter_query \
                   -layers 4 \
                   -rnn_size 256 \
                   -word_vec_size 256 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type qa3 \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 100000 \
                   -valid_steps 5000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 > info/pairenc/5ref_know_fromknow40k &    

    
    

                   # -gpuid 0 \
                   # -report_every 500 \

                   #-log_file logs/$model_name/temp.log \
                   #-tensorboard \
                   #-tensorboard_log_dir logs/$model_name/tensorboard
