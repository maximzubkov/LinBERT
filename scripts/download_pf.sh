#!/bin/bash


DEV=false
DATA_DIR=./data
DATASET_NAME=pf_6_full
X_SHAPE=100
Y_SHAPE=100

function is_int(){
  if [[ ! "$1" =~ ^[+-]?[0-9]+$ ]]; then
    echo "Non integer {$1} passed in --$2"
    exit 1
  fi
}

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset=NAME             specify dataset name, available: pf_6_full, pf_9_full, pf_14_full"
      echo "-x_shape / y_shape             specify x / y shape of image after resizing"
      echo "--dev                          pass it if developer mode should be used, default false"
      exit 0
      ;;
    -d|--dataset*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        DATASET_NAME=$2
        shift 2
      else
        echo "Specify dataset name"
        exit 1
      fi
      ;;
    --x_shape*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        is_int "$2" "x_shape"
        X_SHAPE=$2
        shift 2
      else
        echo "Specify x shape"
        exit 1
      fi
      ;;
    --y_shape*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        is_int "$2" "y_shape"
        Y_SHAPE=$2
        shift 2
      else
        echo "Specify y shape"
        exit 1
      fi
      ;;
    --dev*)
      DEV=true
      shift
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

INDEX=${DATASET_NAME:3:1}

if [ "$INDEX" -eq "6" ]
then
  ZIP_FILE=pf6.zip
  OLD_DATASET_NAME=curv_contour_length_6_full
  LINK="https://drive.google.com/u/0/uc?id=13y1cclbbyqm3pqCtxX0D5F_rBo9s2bKb"
elif [ "$INDEX" -eq "9" ]
then
  ZIP_FILE=pf9.zip
  OLD_DATASET_NAME=curv_contour_length_9_full
  LINK="https://drive.google.com/u/0/uc?id=1ZrZvfWcl1OvUQWxnkQxOdyantpA7NTR8"
elif [ "$INDEX" -eq "1" ]
then
  ZIP_FILE=pf14.zip
  OLD_DATASET_NAME=curv_contour_length_14_full
  LINK="https://drive.google.com/u/0/uc?id=1Qju_FmGQwoOeIZ-Tj8csmniBEG5YqdlN"
fi

if [ ! -f $DATA_DIR/$ZIP_FILE ]
then
  echo "Downloading "$ZIP_FILE
  gdown $LINK
  mv $ZIP_FILE $DATA_DIR/$ZIP_FILE
fi

if $DEV
then
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/train/0.npz
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/train/1.npz
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/val/0.npz
  mv $OLD_DATASET_NAME/val $OLD_DATASET_NAME/test
  mv $OLD_DATASET_NAME "$DATA_DIR"/${DATASET_NAME}_small
  python scripts/preprocess_pf.py --dataset=${DATASET_NAME}_small --x_shape=$X_SHAPE --y_shape=$Y_SHAPE
fi

unzip $DATA_DIR/$ZIP_FILE -d $DATA_DIR
mv $DATA_DIR/$OLD_DATASET_NAME $DATA_DIR/$DATASET_NAME
mv $DATA_DIR/$DATASET_NAME/val $DATA_DIR/$DATASET_NAME/test

python scripts/preprocess_pf.py --dataset=$DATASET_NAME --x_shape=$X_SHAPE --y_shape=$Y_SHAPE