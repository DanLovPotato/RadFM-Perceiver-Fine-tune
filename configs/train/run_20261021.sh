# 正式训练配置（架构改版后重启）：perceiver+fc共用(768->4096,配真正的LLaMA-2-7B-chat)，
# 文字侧adapter(768->4096)，对比学习用32次cosine取平均，不再是mean pool。
# 3张卡数据并行，先跑着观察loss曲线再决定要不要提前停。

cuda_devices="0,1,2"
master_port="29500"

data_root="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
data_folder="$data_root/train_preprocessed"
mask_folder="$data_root/train_region_mask"
csv_file="$data_root/radgenome_files/train_region_report.csv"

output_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261021"

batch_size=4
num_workers=4
lr=1e-4
temperature=0.07
contrastive_weight=0.2
generation_weight=0.8
epochs=10
log_every=20
save_every_steps=1000

smoke_test="false"
no_generation="false"
no_wandb="false"
