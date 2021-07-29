#!/bin/bash

DATA_DIR=data


if [ ! -d "$DATA_DIR" ]
then
    mkdir "$DATA_DIR"
fi

if [ ! -d "$DATA_DIR"/glue ]
then
    mkdir "$DATA_DIR"/glue
fi

python scripts/download_glue_data.py --data_dir "$DATA_DIR"/glue --tasks all