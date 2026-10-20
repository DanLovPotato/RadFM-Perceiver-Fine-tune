#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]
then
    echo "Warning: The configuration file should be specified according to the device being used!"
    exit 1
fi

config_file="$1"
script_name=$(basename "$0" .sh)

# 导入配置文件
source "../configs/${script_name}/${config_file}.sh"

# 激活conda环境（不依赖调用方提前手动source/activate）
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate reg2rg

optional_args=()
if [ -n "${smoke_test:-}" ] && [ "$smoke_test" = "true" ]; then
    optional_args+=(--smoke_test)
fi
if [ -n "${no_generation:-}" ] && [ "$no_generation" = "true" ]; then
    optional_args+=(--no_generation)
fi
if [ -n "${no_wandb:-}" ] && [ "$no_wandb" = "true" ]; then
    optional_args+=(--no_wandb)
fi

# 根据cuda_devices设置的GPU数量自动决定nproc_per_node（照搬Reg2RG自己train_radgenome.sh的做法）
nproc_per_node=$(echo $cuda_devices | tr ',' '\n' | wc -l)

# CUDA_VISIBLE_DEVICES 必须在进程启动时就设好 - 不能靠train.py内部再设置
# (见train.py里的说明：torch的CUDA子系统在import时就做了一些延迟初始化)
# nproc_per_node>1时用torchrun启动多进程数据并行(DistributedDataParallel)；
# =1时直接跑单进程，跟之前冒烟测试完全一样的路径。
if [ "$nproc_per_node" -gt 1 ]; then
    # --master-addr显式指定127.0.0.1，不用默认的localhost——这台机器上localhost会解析
    # 成IPv6导致c10d的socket初始化报错("Address family not supported by protocol")
    CUDA_VISIBLE_DEVICES=$cuda_devices torchrun --nproc_per_node=$nproc_per_node --master-addr=127.0.0.1 --master-port=${master_port:-29500} ../train.py \
        --data_folder "$data_folder" \
        --mask_folder "$mask_folder" \
        --csv_file "$csv_file" \
        --output_dir "$output_dir" \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --lr $lr \
        --temperature $temperature \
        --contrastive_weight $contrastive_weight \
        --generation_weight $generation_weight \
        --epochs $epochs \
        --log_every $log_every \
        --save_every_steps $save_every_steps \
        "${optional_args[@]}"
else
    CUDA_VISIBLE_DEVICES=$cuda_devices python3 ../train.py \
        --data_folder "$data_folder" \
        --mask_folder "$mask_folder" \
        --csv_file "$csv_file" \
        --output_dir "$output_dir" \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --lr $lr \
        --temperature $temperature \
        --contrastive_weight $contrastive_weight \
        --generation_weight $generation_weight \
        --epochs $epochs \
        --log_every $log_every \
        --save_every_steps $save_every_steps \
        "${optional_args[@]}"
fi
