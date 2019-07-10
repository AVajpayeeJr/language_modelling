#!/bin/sh
# $1=languageCode, $2=outputBaseDir, $3=numClasses $4=mixing_lambda


# Trigram Language Model with KN Smoothing
NGRAM_OUT_DIR=$2/$1/ngram
mkdir -p $NGRAM_OUT_DIR

ngram-count -order 3 -kndiscount1 -kndiscount2 -kndiscount3 -interpolate -text $1/train.txt -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/train.txt -debug 0 > $NGRAM_OUT_DIR/3gram_kn_interp.train.out
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/test.txt -debug 0 > $NGRAM_OUT_DIR/3gram_kn_interp.test.out
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/valid.txt -debug 0 > $NGRAM_OUT_DIR/3gram_kn_interp.valid.out


# Interpolated Class Based + Word Trigram Language Model
CLASS_NGRAM_OUT_DIR=$2/$1/class+ngram/$3/$4
mkdir -p $CLASS_NGRAM_OUT_DIR

uniform-classes $1/$3_clusters.txt > $CLASS_NGRAM_OUT_DIR/classes_with_probs.txt

replace-words-with-classes addone=1 normalize=1 outfile=$CLASS_NGRAM_OUT_DIR/class_counts \
classes=$CLASS_NGRAM_OUT_DIR/classes_with_probs.txt $1/train.txt > $CLASS_NGRAM_OUT_DIR/train_replaced.txt

sed 's/ /\n/g' $CLASS_NGRAM_OUT_DIR/train_replaced.txt | sort | uniq > $CLASS_NGRAM_OUT_DIR/vocab_with_class.txt

ngram-count -vocab $CLASS_NGRAM_OUT_DIR/vocab_with_class.txt -order 2 \
-text $CLASS_NGRAM_OUT_DIR/train_replaced.txt -lm $CLASS_NGRAM_OUT_DIR/class_based.srilm

ngram -lm $CLASS_NGRAM_OUT_DIR/class_based.srilm -ppl $1/train.txt -classes $CLASS_NGRAM_OUT_DIR/class_counts


ngram -order 3  -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -mix-lm $CLASS_NGRAM_OUT_DIR/class_based.srilm \
-classes $CLASS_NGRAM_OUT_DIR/class_counts -write-lm $CLASS_NGRAM_OUT_DIR/word_class_3gram.lm -lambda $4

ngram -lm $CLASS_NGRAM_OUT_DIR/word_class_3gram.lm -ppl $1/train.txt -debug 0 > $CLASS_NGRAM_OUT_DIR/word_class_3gram.train.out
ngram -lm $CLASS_NGRAM_OUT_DIR/word_class_3gram.lm -ppl $1/test.txt -debug 0 > $CLASS_NGRAM_OUT_DIR/word_class_3gram.test.out
ngram -lm $CLASS_NGRAM_OUT_DIR/word_class_3gram.lm -ppl $1/valid.txt -debug 0 > $CLASS_NGRAM_OUT_DIR/word_class_3gram.valid.out

