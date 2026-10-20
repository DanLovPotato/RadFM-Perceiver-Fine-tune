# 冒烟测试配置：小batch、单GPU，跑几步确认流水线没问题，不是用来正式训练的

cuda_devices="0"

data_root="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
data_folder="$data_root/train_preprocessed"
mask_folder="$data_root/train_region_mask"
csv_file="$data_root/radgenome_files/train_region_report.csv"

output_dir="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/smoketest"

batch_size=2
num_workers=0
lr=1e-4
temperature=0.07
contrastive_weight=0.2
generation_weight=0.8
epochs=1
log_every=1
save_every_steps=1000

smoke_test="true"
no_generation="false"
