#!/bin/bash

DATA_DIR=./data
if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi
wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt -P $DATA_DIR/
head -100 data/oscar.eo.txt > data/oscar_small.eo.txt