python translate.py -gpu $1 \
     -knowledge \
     -batch_size 20 \
     -beam_size 4 \
     -model models/EMNLP2020-retro/$2 \
     -src haojie/data/dialogs/dialog.test.txt \
     -know inference/EMNLP-retro/knowledge/$3 \
     -output decodings/EMNLP2020-retro/$4 \
     -min_length 5 \
     -max_length 15