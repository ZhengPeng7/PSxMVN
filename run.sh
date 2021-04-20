# cuhk / prw / mvn
dataset="cuhk"

if [ ${dataset} == "cuhk" ]; then epoch="19"; elif [ ${dataset} == "prw" ]; then epoch="17"; else epoch="4"; fi;

# train
python train.py --cfg configs/${dataset}.yaml

# test - vanilla
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth

# test - CBGM
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True

# test - using GT bbox
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_GT True

# test - CBGM, using GT bbox
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True EVAL_USE_GT True
