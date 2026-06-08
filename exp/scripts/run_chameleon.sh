#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=True \
    --d=1 \
    --dataset=chameleon \
    --dropout=0 \
    --early_stopping=100 \
    --epochs=300 \
    --folds=1 \
    --hidden_channels=32 \
    --input_dropout=0.7 \
    --layers=3 \
    --lr=0.01 \
    --model=DiagSheaf \
    --second_linear=True \
    --sheaf_decay=0.0012638885974822734 \
    --weight_decay=0.0002969905682317406 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=False \
    --stop_strategy=acc \
    --entity="${ENTITY}" 