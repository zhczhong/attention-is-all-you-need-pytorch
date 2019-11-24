python preprocess.py -train_src data/cleaned.train.diff -train_tgt data/cleaned.train.msg -valid_src data/cleaned.valid.diff -valid_tgt data/cleaned.valid.msg -save_data data/vocab -max_len 400 -min_word_count 0 -share_vocab
python train.py -data data/vocab -save_model exp/model/ -log exp/log/ -save_mode best  -proj_share_weight -embs_share_weight -label_smoothing -epoch 100

