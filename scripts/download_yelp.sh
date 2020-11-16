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

if [ ! -f "$DATA_DIR/yelp_review_polarity_csv.tgz" ]
then
  wget -c https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz -P $DATA_DIR/
fi

tar -C $DATA_DIR/ -xvf "$DATA_DIR/yelp_review_polarity_csv.tgz"
mv $DATA_DIR/yelp_review_polarity_csv $DATA_DIR/yelp_polarity
