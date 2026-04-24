#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=False \
    --d=2 \
    --dataset=citeseer \
    --dropout=0.2 \
    --early_stopping=300 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=16 \
    --input_dropout=0.7 \
    --layers=2 \
    --lr=0.01 \
    --model=BundleSheaf \
    --second_linear=True \
    --sheaf_decay=0.0012638885974822734 \
    --weight_decay=0.0002969905682317406 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 