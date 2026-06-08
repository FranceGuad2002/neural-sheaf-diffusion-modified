#!/bin/sh

python -m exp.run \
    --add_lp=False \
    --d=1 \
    --dataset=squirrel \
    --dropout=0 \
    --early_stopping=100 \
    --epochs=300 \
    --folds=1 \
    --hidden_channels=32 \
    --input_dropout=0.7 \
    --layers=3 \
    --lr=0.01 \
    --model=DiagSheaf \
    --orth=householder \
    --second_linear=True \
    --weight_decay=0.00011215791366362148 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 