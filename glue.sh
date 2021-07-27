#!/bin/bash

DATASET_NAME=mnli
N_EPOCHS=3
SEED=42
IS_LINEAR=false
FEATURE_MAP=""
PB_TYPE=""
BIAS_TYPE=""

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset NAME             specify dataset name from glue benchmark"
      echo "--seed INT                 specify seed value"
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
    --is_linear*)
      IS_LINEAR=true
      shift
      ;;
    --feature_map*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        FEATURE_MAP=$2
        shift 2
      else
        echo "Specify feature_map"
        exit 1
      fi
      ;;
    --pos_bias*)
      PB_TYPE=fft
      BIAS_TYPE=full
      shift
      ;;
    --seed*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        SEED=$2
        shift 2
      else
        echo "Specify seed"
        exit 1
      fi
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

OUTPUT_DIR=/tmp/"$DATASET_NAME"-"$SEED"

if $IS_LINEAR
then
    OUTPUT_DIR="$OUTPUT_DIR"-lin-"$FEATURE_MAP"
else
    OUTPUT_DIR="$OUTPUT_DIR"-orig
fi

if [ "$PB_TYPE" = "fft" ]
then
    OUTPUT_DIR="$OUTPUT_DIR"-fft
fi

if [ "$DATASET_NAME" = "mrpc" ] || [ "$DATASET_NAME" = "wnli" ]
then
    N_EPOCHS=5
fi

python glue.py \
  --model_name_or_path bert-base-cased \
  --task_name "$DATASET_NAME" \
  --do_train \
  --do_eval \
  --seed $SEED \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $N_EPOCHS \
  --output_dir /tmp/"$DATASET_NAME"-"$SEED" \
  --is_linear "$IS_LINEAR" \
  --feature_map "$FEATURE_MAP" \
  --pos_bias_type "$PB_TYPE" \
  --bias_base_type "$BIAS_TYPE"