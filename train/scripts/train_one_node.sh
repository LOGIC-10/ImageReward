#! /bin/bash

# 切换到指定路径
cd /bjzhyai03/workhome/luoqinyu/WorkSpace/ImageReward/train || exit

NUM_GPUS_PER_WORKER=2
MASTER_PORT=29527

# 获取当前时间戳，格式为 YYYYMMDD_HHMM
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# 检查 Comment 是否为空，如果为空则不包含它，否则包含它
Comment='OnlineSource原始组合冻住零点九'

if [ -z "$Comment" ]; then
    SAVEPATH="${TIMESTAMP}_blip_uni_cross_mul"
else
    SAVEPATH="${TIMESTAMP}_${Comment}_blip_uni_cross_mul"
fi

# PRE_LOAD="/bjzhyai03/workhome/luoqinyu/WorkSpace/ImageReward/train/checkpoint_base/20241014_0626_blip_uni_cross_mul_bs512_fix=0.7_lr=0.001cosine/best_lr=0.001.pt"


train_options=" \
       --savepath ${SAVEPATH} \
       --batch_size 32 \
       --accumulation_steps 2 \
       --epochs 30 \
       --distributed True \
       --gpu_num ${NUM_GPUS_PER_WORKER} \
       --gpu_id '2,3' \
       --clear_visualizer \
       --fix_rate 0.9 \
       --lr 1e-03 \
       --lr-decay-style cosine \
       --warmup 0.0 \
       --rank_pair \
       --load_pair_store \
       --std_log \
       --valid_per_epoch 4 \
"

run_cmd="torchrun
     --nnodes=1
     --nproc_per_node=${NUM_GPUS_PER_WORKER}
     --master_port=${MASTER_PORT}
     /bjzhyai03/workhome/luoqinyu/WorkSpace/ImageReward/train/src/train.py ${train_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x