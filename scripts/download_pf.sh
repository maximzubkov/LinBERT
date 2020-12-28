#!/bin/bash


DEV=false
DATA_DIR=./data
DATASET_NAME=pf_6_full

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset=NAME             specify dataset name, available: pf_6_full, pf_9_full, pf_14_full"
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
  LINK="https://drive.google.com/u/0/uc?id=13y1cclbbyqm3pqCtxX0D5F_rBo9s2bKb"
elif [ "$INDEX" -eq "9" ]
then
  ZIP_FILE=pf9.zip
  LINK="https://drive.google.com/u/0/uc?id=1ZrZvfWcl1OvUQWxnkQxOdyantpA7NTR8"
elif [ "$INDEX" -eq "1" ]
then
  ZIP_FILE=pf14.zip
  LINK="https://drive.google.com/u/0/uc?id=1Qju_FmGQwoOeIZ-Tj8csmniBEG5YqdlN"
fi

if [ ! -f $DATA_DIR/$ZIP_FILE ]
then
  echo "Downloading "$ZIP_FILE
  gdown $LINK
  mv $ZIP_FILE $DATA_DIR/$ZIP_FILE
fi

OLD_DATASET_NAME=curv_contour_length_${INDEX}_full
if $DEV
then
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/train/0.npz
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/train/1.npz
  unzip $DATA_DIR/$ZIP_FILE $OLD_DATASET_NAME/train/2.npz
  mkdir $OLD_DATASET_NAME/test
  mv $OLD_DATASET_NAME/train/2.npz $OLD_DATASET_NAME/test/0.npz
  mv $OLD_DATASET_NAME "$DATA_DIR"/${DATASET_NAME}_small
  python scripts/preprocess_pf.py --dataset ${DATASET_NAME}_small
fi

unzip $DATA_DIR/$ZIP_FILE -d $DATA_DIR
mv $DATA_DIR/$OLD_DATASET_NAME $DATA_DIR/$DATASET_NAME

python preprocess_pf.py --dataset $DATASET_NAME