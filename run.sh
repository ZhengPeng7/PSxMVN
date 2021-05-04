# cuhk / prw / mvn
dataset="mvn"

# Keep consistant with N in sub_prw.sh
NGPU=1

if [ ${dataset} == "cuhk" ]; then let epoch=20*${NGPU}-1; elif [ ${dataset} == prw ]; then let epoch=18*${NGPU}-1; else let epoch=10*${NGPU}-1; fi;

# Train
python -m torch.distributed.launch --nproc_per_node=${NGPU} --use_env train.py --cfg configs/${dataset}.yaml --world-size ${NGPU}

# Test

# vanilla
# let ep_st=epoch/2
# for (( ep = ${ep_st}; ep < ${epoch}; ep ++ ))
# do
# python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth
# # done

# # CBGM
# python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True

# # using GT bbox
# python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_GT True

# # CBGM, using GT bbox
# python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True EVAL_USE_GT True
