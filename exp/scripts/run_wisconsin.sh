#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=False \
    --d=1 \
    --dataset=wisconsin \
    --dropout=0.7276458263736642 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=16 \
    --input_dropout=0 \
    --layers=2 \
    --lr=0.02 \
    --model=DiagSheaf \
    --orth=householder \
    --sheaf_act=tanh \
    --weight_decay=0.0006685729356079199 \
    --use_act=True \
    --normalised=True \
    --sparse_learner=False \
    --edge_weights=True \
    --entity="${ENTITY}"