#!/bin/bash

DATA_DIR=./data
if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

if [ ! -f "$DATA_DIR/train-images-idx3-ubyte.gz" ]
then
  wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P $DATA_DIR/
fi

if [ ! -f "$DATA_DIR/train-labels-idx1-ubyte.gz" ]
then
  wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P $DATA_DIR/
fi

if [ ! -f "$DATA_DIR/t10k-images-idx3-ubyte.gz" ]
then
  wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P $DATA_DIR/
fi

if [ ! -f "$DATA_DIR/t10k-labels-idx1-ubyte.gz" ]
then
  wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P $DATA_DIR/
fi

if [ ! -d "$DATA_DIR/mnist" ]
then
  mkdir $DATA_DIR/mnist
fi

gunzip -c "$DATA_DIR/train-images-idx3-ubyte" > $DATA_DIR/mnist/train_images
gunzip -c "$DATA_DIR/train-labels-idx1-ubyte" > $DATA_DIR/mnist/train_labels
gunzip -c "$DATA_DIR/t10k-images-idx3-ubyte" > $DATA_DIR/mnist/test_images
gunzip -c "$DATA_DIR/t10k-labels-idx1-ubyte" > $DATA_DIR/mnist/test_labels

python scripts/preprocess_mnist.py --dataset="${DATA_DIR}/mnist"

if [ ! -d "$DATA_DIR/mnist_small" ]
then
  mkdir $DATA_DIR/mnist_small
fi

head -300 $DATA_DIR/mnist/test.csv > $DATA_DIR/mnist_small/test.csv
head -1000 $DATA_DIR/mnist/train.csv > $DATA_DIR/mnist_small/train.csv
