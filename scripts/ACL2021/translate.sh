python translate.py -gpu $1 \
     -knowledge \
     -batch_size 20 \
     -beam_size 4 \
     -model models/EMNLP2020-retro/$1 \
     -src haojie/data/dialogs/dialog.test.txt \
     -know inference/EMNLP-retro/knowledge/$2 \
     -output decodings/EMNLP2020-retro/esa_v1_trained_with_one_cap_potentially/top5_45k \
     -min_length 5 \
     -max_length 15 \
     -verbose  