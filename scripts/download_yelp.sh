#!/bin/bash

DATA_DIR=./data
if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

if [ ! -f "$DATA_DIR/yelp_review_full_csv.tgz" ]
then
  wget -c https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz -P $DATA_DIR/
fi

tar -C $DATA_DIR/ -xvf "$DATA_DIR/yelp_review_full_csv.tgz"
mv $DATA_DIR/yelp_review_full_csv $DATA_DIR/yelp_full
printf '%s\n' "label,text" | cat - data/yelp_full/test.csv > tmp && mv tmp data/yelp_full/test.csv
printf '%s\n' "label,text" | cat - data/yelp_full/train.csv > tmp && mv tmp data/yelp_full/train.csv

head -1000 $DATA_DIR/yelp_full/test.csv > $DATA_DIR/yelp_full/test_small.csv
head -300 $DATA_DIR/yelp_full/train.csv > $DATA_DIR/yelp_full/train_small.csv

if [ ! -f "$DATA_DIR/yelp_review_polarity_csv.tgz" ]
then
  wget -c https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz -P $DATA_DIR/
fi

tar -C $DATA_DIR/ -xvf "$DATA_DIR/yelp_review_polarity_csv.tgz"
mv $DATA_DIR/yelp_review_polarity_csv $DATA_DIR/yelp_polarity
printf '%s\n' "label,text" | cat - data/yelp_polarity/test.csv > tmp && mv tmp data/yelp_polarity/test.csv
printf '%s\n' "label,text" | cat - data/yelp_polarity/train.csv > tmp && mv tmp data/yelp_polarity/train.csv

head -1000 $DATA_DIR/yelp_polarity/test.csv > $DATA_DIR/yelp_polarity/test_small.csv
head -300 $DATA_DIR/yelp_polarity/train.csv > $DATA_DIR/yelp_polarity/train_small.csv
