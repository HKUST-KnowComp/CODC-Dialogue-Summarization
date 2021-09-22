python translate.py -gpu 2 \
     -batch_size 20 \
     -beam_size 4 \
     -model models/pairenc/5ref_from_100k_step_160000.pt \
     -src haojie/data/dialogs/dialog.test.txt \
     -output results/pairenc/5ref_160k \
     -min_length 5 \
     -max_length 15 \
     -pairenc \
     -verbose 