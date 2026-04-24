#!/bin/sh

python -m exp.run \
    --dataset=texas \
    --d=1 \
    --layers=2 \
    --hidden_channels=16 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.02 \
    --epochs=500 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --folds=10 \
    --model=DiagSheaf \
    --normalised=True \
    --sparse_learner=True \
    --entity="${ENTITY}"