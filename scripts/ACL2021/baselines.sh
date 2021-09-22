CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python -u train.py \
    -save_model models/EMNLP2020-retro/fivecap_baselines/trans_copy \
    -data dataset/EMNLP2020-retro/train_haojie \
    -layers 4 \
    -rnn_size 256 \
    -word_vec_size 256 \
    -max_grad_norm 0 \
    -optim adam \
    -encoder_type transformer \
    -decoder_type transformer \
    -position_encoding \
    -dropout 0.2 \
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
    -train_steps 50000 \
    -accum_count 4 \
    -share_embeddings \
    -param_init_glorot \
    -copy_attn \
    -reuse_copy_attn \
    -world_size 1 \
    -gpu_ranks 0 \
    -valid_steps 2500 > logs/EMNLP-retro/trans_copy &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python -u train.py \
    -save_model models/EMNLP2020-retro/fivecap_baselines/copy_attn \
    -data dataset/EMNLP2020-retro/train_haojie  \
    -copy_attn -global_attention mlp \
    -word_vec_size 128 -rnn_size 256 -layers 1 \
    -encoder_type brnn -train_steps 120000 \
    -max_grad_norm 2 -dropout 0. -batch_size 16 -valid_batch_size 16 \
    -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 \
    -reuse_copy_attn -copy_loss_by_seqlength -bridge \
    -seed 229 -world_size 1 -gpu_ranks 0 -valid_steps 10000 > logs/EMNLP-retro/copy_attn &    






