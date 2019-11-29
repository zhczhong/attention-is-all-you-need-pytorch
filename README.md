# Commit message generation with Transformer

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Requirement
- python 3.4+
- pytorch 0.4.1+
- tqdm
- numpy

# Pretraining
### 0) Download the data.
Download and unzip the dataset into MASS-unsupNMT/data/processed/en-fr.

### 1) Preprocess the data
```
mv cleaned.test.diff test.en
mv cleaned.test.msg test.fr
mv cleaned.train.diff train.en
mv cleaned.train.msg test.fr
mv cleaned.valid.diff valid.en
mv cleaned.valid.msg test.fr
```
And then build 3 vocabulary file, vocab.en, vocab.en-fr, vocab.fr in MASS-unsupNMT/data/processed/en-fr. The format of vocabulary is:
```
word1 frequency
word2 frequency
...
wordn frequency
```

And run the following command in MASS-unsupNMT directory:
```
SRC="en"
TGT="fr"

MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT

# raw and tokenized files
SRC_RAW=$MONO_PATH/$SRC/all.$SRC
TGT_RAW=$MONO_PATH/$TGT/all.$TGT
SRC_TOK=$SRC_RAW.tok
TGT_TOK=$TGT_RAW.tok

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

# train / valid / test monolingual BPE data
SRC_TRAIN_BPE=$PROC_PATH/train.$SRC
TGT_TRAIN_BPE=$PROC_PATH/train.$TGT
SRC_VALID_BPE=$PROC_PATH/valid.$SRC
TGT_VALID_BPE=$PROC_PATH/valid.$TGT
SRC_TEST_BPE=$PROC_PATH/test.$SRC
TGT_TEST_BPE=$PROC_PATH/test.$TGT

# valid / test parallel BPE data
PARA_SRC_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$TGT

$MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_TRAIN_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_TRAIN_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TEST_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TEST_BPE

ln -sf $PARA_SRC_VALID_BPE.pth $SRC_VALID_BPE.pth
ln -sf $PARA_TGT_VALID_BPE.pth $TGT_VALID_BPE.pth
ln -sf $PARA_SRC_TEST_BPE.pth  $SRC_TEST_BPE.pth
ln -sf $PARA_TGT_TEST_BPE.pth  $TGT_TEST_BPE.pth
```
### 2) Pre-training
```
python train.py                                      \
--exp_name unsupMT_enfr                              \
--data_path ./data/processed/en-fr/                  \
--lgs 'en-fr'                                        \
--mass_steps 'en,fr'                                 \
--encoder_only false                                 \
--emb_dim 512                                       \
--n_layers 2                                         \
--n_heads 8                                          \
--dropout 0.1                                        \
--attention_dropout 0.1                              \
--gelu_activation true                               \
--tokens_per_batch 3000                              \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 5000                                  \
--max_epoch 500                                      \
--eval_bleu true                                     \
--word_mass 0.5                                      \
--min_len 5                                          \
```

# Usage
### 0) Download the data.
Download and unzip the dataset into the folder.

### 1) Preprocess the data.
```bash
python preprocess.py -train_src pathtodata/train_sourcefile -train_tgt pathtodata/train_targetfile -valid_src pathtodata/valid_sourcefile -valid_tgt pathtodata/valid_targetfile -save_data pathtodata/vocab -max_len 400 -min_word_count 0 -share_vocab
```

### 2) Train the model
```bash
python train.py -data pathtodata/vocab -save_model exp/model/ -log exp/log/ -save_mode best  -proj_share_weight -embs_share_weight -label_smoothing -epoch 100
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding. 

### 3) Test the model
```bash
python translate.py -model exp/model/trained.chkpt -vocab pathtodata/vocab -src pathtodata/test_sourcefile -output exp/resultfile
```
### 4) Evaluate the result
```bash
python evaluate/evaluate.py pathto/candidate pathto/reference
perl evaluate/multi-bleu.perl pathto/reference < pathto/candidate
```
---
# Performance

---
# TODO

---
# Acknowledgement

