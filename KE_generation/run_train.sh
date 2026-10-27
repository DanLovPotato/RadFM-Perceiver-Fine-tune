#!/bin/bash
# 一步生成 train 用的全部 knowledge bank 文件（流程见同目录 README.md）：
#   $out_dir/unique_reports.json                      [0]  train 报告按器官去重
#   $out_dir/bank_labels.json                         [0b] 每句话正常/异常
#   $out_dir/organ_report_embeddings_unique_4096.npz  [1]  去重文本的 4096 维向量
#   $out_dir/organ_report_embeddings.npz              [2]  ★ Reg2RG 的 bank_npy_path（train/val 共用）
#   $out_dir/prototypes.npz                           [1b] 正常/异常原型
#   $out_dir/train/image_mean.npz                     [3]  train 图像均值，异常池去均值用（run_test.sh 要用）
#   $out_dir/train/organ_annotation.json              [3]  ★ Reg2RG 的 organ_annotation_path（train）
#   $out_dir/train/decisions.npz                      [3]  每个病人每个器官的零样本判断（检查用；不做判断的 4 个器官没有）
#   $out_dir/train/image_queries.pt                   [3]  图像 query 缓存，重跑检索时直接复用
# 前面几步的输出已经存在就跳过。run_test.sh 复用这里的文件，所以必须先跑完这个脚本。
# out_dir 按 ckpt_name 区分，换了 checkpoint 就是新目录。
# 用法：screen -dmS ke_train bash -c 'bash run_train.sh 2>&1 | tee ke_train.log'
set -euo pipefail

cuda_device="1"
ckpt_name="step145000"   # adapter / perceiver / fc 都用这个 step，必须配套
ckpt_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261021"
out_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset/KE_bank_4096_${ckpt_name}"
bank_dtype="float32"     # float16 可以把 bank 从 ~4GB 减到 ~2GB，形状不变
batch_size=2       # 每批几个病人（每个病人最多 10 个器官）
num_workers=16     # 读 CT 的 CPU 进程数；瓶颈在这里，不在 GPU

# 激活 conda 环境（不依赖调用方提前手动 source/activate）
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate reg2rg

cd "$(dirname "$0")"
unique_json="$out_dir/unique_reports.json"
unique_npz="$out_dir/organ_report_embeddings_unique_4096.npz"
dense_npz="$out_dir/organ_report_embeddings.npz"
labels_json="$out_dir/bank_labels.json"

# CUDA_VISIBLE_DEVICES 必须在进程启动时就设好（见 train.py 里的说明）
if [ ! -f "$unique_json" ]; then
    echo "=== [0/3] unique reports ==="
    python3 get_unique_reports.py --output_path "$unique_json"
else
    echo "=== [0/3] unique reports: $unique_json exists, skip ==="
fi

if [ ! -f "$labels_json" ]; then
    echo "=== [0b/3] bank labels (normal / abnormal) ==="
    python3 label_bank.py --unique_reports "$unique_json" --output_path "$labels_json"
else
    echo "=== [0b/3] bank labels: $labels_json exists, skip ==="
fi

if [ ! -f "$unique_npz" ]; then
    echo "=== [1/3] text embeddings ==="
    CUDA_VISIBLE_DEVICES=$cuda_device python3 compute_text_embeddings_4096.py \
        --unique_reports "$unique_json" \
        --adapter_checkpoint "$ckpt_dir/adapter-${ckpt_name}.pth" \
        --output_path "$unique_npz"
else
    echo "=== [1/3] text embeddings: $unique_npz exists, skip ==="
fi

if [ ! -f "$dense_npz" ]; then
    echo "=== [2/3] dense bank ==="
    python3 build_dense_bank.py \
        --unique_reports "$unique_json" \
        --unique_feats "$unique_npz" \
        --output_path "$dense_npz" \
        --dtype $bank_dtype
else
    echo "=== [2/3] dense bank: $dense_npz exists, skip ==="
fi

echo "=== [3/3] train organ_annotation.json ==="
CUDA_VISIBLE_DEVICES=$cuda_device python3 build_organ_annotation.py \
    --split train \
    --ckpt_dir "$ckpt_dir" \
    --ckpt_name "$ckpt_name" \
    --unique_reports "$unique_json" \
    --unique_feats "$unique_npz" \
    --bank_labels "$labels_json" \
    --prototypes "$out_dir/prototypes.npz" \
    --image_mean "$out_dir/train/image_mean.npz" \
    --decisions_out "$out_dir/train/decisions.npz" \
    --output_path "$out_dir/train/organ_annotation.json" \
    --query_cache "$out_dir/train/image_queries.pt" \
    --batch_size $batch_size \
    --num_workers $num_workers

echo "Done. Reg2RG train config:"
echo "  bank_npy_path=$dense_npz"
echo "  organ_annotation_path=$out_dir/train/organ_annotation.json"
