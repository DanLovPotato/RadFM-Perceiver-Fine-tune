#!/bin/bash
# 一步生成 validation 用的 knowledge bank 文件（流程见同目录 README.md）：
#   $out_dir/val/organ_annotation.json                [3] ★ Reg2RG 的 organ_annotation_path（validation）
#   $out_dir/val/decisions.npz                        [3] 每个病人每个器官的零样本判断（检查用；不做判断的 4 个器官没有）
#   $out_dir/val/image_queries.pt                     [3] 图像 query 缓存
# bank_npy_path 用 run_train.sh 生成的同一个 organ_report_embeddings.npz。
# validation 没有自己的文本库：val 图像检索的是 train 的同一份文本库，邻居全是 train 病人。
# 必须先跑完 run_train.sh：这里复用它的 [0][0b][1][1b][2] 输出，还要用它算出的
# train 图像均值（train/image_mean.npz）。设置要跟 run_train.sh 保持一致，尤其是 ckpt_name。
# 用法：screen -dmS ke_test bash -c 'bash run_test.sh 2>&1 | tee ke_test.log'
set -euo pipefail

cuda_device="2"
ckpt_name="step145000"   # adapter / perceiver / fc 都用这个 step，必须跟 run_train.sh 一致
ckpt_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261021"
out_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset/KE_bank_4096_${ckpt_name}"
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

for f in "$unique_json" "$labels_json" "$unique_npz" "$dense_npz" "$out_dir/prototypes.npz" "$out_dir/train/image_mean.npz"; do
    if [ ! -f "$f" ]; then
        echo "Missing $f - run run_train.sh (same ckpt_name) to the end first."
        exit 1
    fi
done

# CUDA_VISIBLE_DEVICES 必须在进程启动时就设好（见 train.py 里的说明）
echo "=== [3/3] validation organ_annotation.json ==="
CUDA_VISIBLE_DEVICES=$cuda_device python3 build_organ_annotation.py \
    --split val \
    --ckpt_dir "$ckpt_dir" \
    --ckpt_name "$ckpt_name" \
    --unique_reports "$unique_json" \
    --unique_feats "$unique_npz" \
    --bank_labels "$labels_json" \
    --prototypes "$out_dir/prototypes.npz" \
    --image_mean "$out_dir/train/image_mean.npz" \
    --decisions_out "$out_dir/val/decisions.npz" \
    --output_path "$out_dir/val/organ_annotation.json" \
    --query_cache "$out_dir/val/image_queries.pt" \
    --batch_size $batch_size \
    --num_workers $num_workers

echo "Done. Reg2RG validation config:"
echo "  bank_npy_path=$dense_npz"
echo "  organ_annotation_path=$out_dir/val/organ_annotation.json"
