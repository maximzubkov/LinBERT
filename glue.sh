#!/bin/bash

DATASET_NAME=mnli
N_EPOCHS=3

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset NAME             specify dataset name, available: codeforces, poj_104"
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
    *)
      echo "something went wrong"
      exit 1
  esac
done

if [ "$DATASET_NAME" = "mrpc" ] || [ "$DATASET_NAME" = "wnli" ]
then
    N_EPOCHS=5
fi

python glue.py \
  --model_name_or_path bert-base-cased \
  --task_name "$DATASET_NAME" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $N_EPOCHS \
  --output_dir /tmp/"$DATASET_NAME"