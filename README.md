# radfm_perceiver_align

训练RadFM的Perceiver模块（+配套的fc投影层），目标是把训好的`perceiver`和`fc`权重直接加载进Reg2RG的`self.perceiver`/`self.fc`，替换掉原来的版本。

## 模型结构

```
                    3D CT图像 volume
                         ↓
              ┌──────────────────────┐
              │ vision_encoder (ViT)  │ ← 冻住
              └──────────┬───────────┘
                         ↓ patch_tokens (768维)
              ┌──────────────────────┐
              │   perceiver            │ ← 训练①（最终要搬进Reg2RG）
              └──────────┬───────────┘
                         ↓ 32个token (768维)
              ┌──────────────────────┐
              │   fc (768→4096)        │ ← 训练②（两条分支共用，最终也要搬进Reg2RG）
              └──────────┬───────────┘
                         ↓ 32个token (4096维，不pool，原样往下分两路)
            ┌────────────┴────────────┐
            ↓                         ↓
      ┌─────────────┐          拼进文字embedding前面
      │   对比学习    │                 ↓
      │ (32次cosine   │         ┌───────────────────┐
      │  算完求平均)   │         │ LLaMA-2-7B-chat(冻) │
      └──────┬──────┘          └─────────┬─────────┘
             ↑                           ↓
             │                   next-token交叉熵loss
             │                      （生成loss）
      ┌──────┴──────┐
      │   adapter    │ ← 训练③（只给对比学习用，不进Reg2RG）
      │ (768→4096)   │
      └──────┬──────┘
             ↑ [CLS](768维)→1个向量(4096维)
      ┌──────┴──────┐
      │ frozen BERT  │ ← 冻住（fvlm微调过的BiomedVLP-CXR-BERT）
      └──────┬──────┘
             ↑
          文字report
```

- **两条loss**：对比学习loss（图像token跟文字向量做32次cosine再取平均）+ 生成loss（图像token接LLaMA-2-7B-chat做next-token预测），按`contrastive_weight:generation_weight`（默认0.2:0.8）加权合并。
- **梯度归一化**（`grad_balance.py`）：perceiver和fc是两条loss共享的参数，各自loss的梯度先归一化到单位范数再按权重合并，避免某条loss的梯度量级过大把另一条"淹没"；adapter只被对比学习用到，直接拿对比loss的梯度即可。
- **4096这个维度**不是随便选的，是Reg2RG实际用的LLaMA-2-7B-chat的hidden size（不是RadFM原生13B的5120），这样fc训完才能直接搬进Reg2RG的`self.fc`。

## 依赖

**代码**：
- 这个仓库本身，不再需要fvlm、RadFM仓库的代码——`PerceiverResampler`、`ViT`、BERT环境补丁这三块代码复制进了`vendored.py`，第三方库`lavis`（Salesforce开源，fvlm只是vendor了一份，不是fvlm自己写的）整个文件夹复制进了本仓库的`lavis/`目录，都不再live import，**fvlm仓库不需要了**。
- 仍然需要 **Reg2RG archive**（`Reg2RG-debug-current-full_correct_F_20260920_archive`，放在跟本仓库同级目录）：`dataset.py`依赖它的`RadGenomeDataset_Train`，故意保持跟Reg2RG完全一致的CT预处理逻辑，不vendor。

**权重文件**（路径写死在`model.py`/`generation.py`里，按需改）：
| 文件 | 作用 |
|---|---|
| `RadFM_vit3d.pth` | vision_encoder（冻住）|
| `RadFM_perceiver_fc.pth` | 只用来给perceiver热启动（文件里虽然也有`fc`这个key，但我们故意不加载它——fc是随机初始化、自己从头训练的，跟Reg2RG自己的precedent一致，见`model.py`里`build_fc()`的说明）|
| `model.pth` | 冻住的BERT文字编码器权重，fvlm官方公开发布（https://drive.google.com/drive/folders/15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD ）|
| `llama_weights/`（LLaMA-2-7B-chat-hf） | 生成分支的冻住decoder |

**环境**：conda环境`reg2rg`（本项目一直在这个环境里跑），关键依赖包括`torch`、`transformers`、`monai`、`lavis`（走fvlm那份，不是pip装的）、`wandb`（可选）。

## 怎么跑

### 1. 冒烟测试（先确认代码能跑通，不是正式训练）

```bash
cd scripts
bash train.sh train   # 对应 configs/train/train.sh，单GPU、batch_size=2、跑3步就退出
```

### 2. 正式训练

```bash
cd scripts
bash train.sh run_20261021   # 对应 configs/train/run_20261021.sh，3张卡、10个epoch
```

配置文件里能改的主要参数：`cuda_devices`（用哪几张卡）、`batch_size`、`lr`、`contrastive_weight`/`generation_weight`、`epochs`、`save_every_steps`。改配置就复制一份`configs/train/*.sh`，改好后用`bash train.sh <你的配置文件名（不带.sh）>`启动。

训练日志会打印在screen里（建议用`screen -dmS <名字> bash -c 'bash train.sh <配置> 2>&1 | tee <日志路径>'`这样跑，断开终端也不会中断）。如果配置里`no_wandb="false"`，还能在wandb看曲线。

### 3. checkpoint怎么存的

每`save_every_steps`步，`perceiver-stepX.pth`、`fc-stepX.pth`、`adapter-stepX.pth`三个一起存到`output_dir`；训练结束再存一次`*-final.pth`。真正要搬进Reg2RG的只有`perceiver`和`fc`这两个，`adapter`只是训练阶段用，不进Reg2RG。

### 4. 训完之后：生成knowledge bank（4096维）

全部代码在`KE_generation/`里，流程图和说明见[KE_generation/README.md](KE_generation/README.md)。改好脚本开头的`cuda_device`/`ckpt_name`后：

```bash
cd KE_generation
bash run_train.sh   # train 的 bank + organ_annotation.json
bash run_test.sh    # validation 的 organ_annotation.json（bank 跟 train 共用）
```

## 已知限制

- `diag_gradient_scale.py`是早期架构（mean pool + 共享维度不是4096）时写的一次性诊断脚本，已经完成它的历史使命（定位梯度不平衡问题），没有跟着后面的改动同步更新，现在跑不了，仅供参考代码逻辑。
- `eval_retrieval.py`、`eval_generation.py`已经按当前架构（perceiver+fc共用4096维、32次cosine平均、真正的LLaMA-2-7B-chat）更新过，可以正常跑，用于在embed进Reg2RG之前快速比较候选checkpoint。
